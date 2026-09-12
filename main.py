import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import yt_dlp

from video.source import (
    validate_source_video,
    get_duration,
    extract_audio,
)

from audio.whisper import transcribe_with_language
from ai.clip_selector import select_clips
from video.clip_renderer import render_clips


# ============================================================
# CONFIG
# ============================================================

SOURCE_VIDEO = Path("input/source.mp4")
AUDIO_FILE = Path("output/source_audio.wav")
CLIPS_DIR = Path("output/clips")

RUTUBE_CHANNEL_URL = "https://rutube.ru/channel/23968031/"

# Максимальное количество видео, которые проверяем
RUTUBE_MAX_VIDEOS = 20

# Минимальная длительность исходного видео
# 20 минут = 1200 секунд
MIN_SOURCE_DURATION = 20 * 60

# Максимальное количество клипов
MAX_CLIPS = 3

# Максимальная высота исходного видео
MAX_VIDEO_HEIGHT = 720

# Таймаут скачивания одного видео
DOWNLOAD_TIMEOUT = 25 * 60

# Сколько секунд ждать между попытками
RETRY_DELAY = 3


# ============================================================
# HELPERS
# ============================================================

def parse_duration(value):
    """
    Преобразует длительность в секунды.
    Поддерживает:
      - int / float
      - строки HH:MM:SS
      - строки MM:SS
    """

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        pass

    parts = value.split(":")

    try:
        parts = [int(x) for x in parts]
    except ValueError:
        return None

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    if len(parts) == 1:
        return float(parts[0])

    return None


def format_duration(seconds):
    if seconds is None:
        return "unknown"

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def clean_source_file():
    """
    Удаляем старый source.mp4 перед новым скачиванием.
    """

    SOURCE_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    if SOURCE_VIDEO.exists():
        print("🗑 Removing old source.mp4")
        SOURCE_VIDEO.unlink()


def print_video_info(video, index=None):
    title = video.get("title") or "Unknown title"
    video_id = video.get("id") or "unknown"
    duration = parse_duration(video.get("duration"))
    url = video.get("webpage_url") or video.get("url") or ""

    prefix = f"[{index}] " if index is not None else ""

    print(
        f"{prefix}{title}\n"
        f"    ID: {video_id}\n"
        f"    Duration: {format_duration(duration)}\n"
        f"    URL: {url}"
    )


# ============================================================
# RUTUBE
# ============================================================

def get_rutube_videos():
    """
    Получает список видео с RUTUBE-канала.

    Используется yt-dlp как extractor:
        https://rutube.ru/channel/23968031/

    Канал поддерживается актуальным yt-dlp через RutubeChannelIE.
    """

    print("=" * 70)
    print("📺 RUTUBE")
    print("=" * 70)

    print(f"Channel: {RUTUBE_CHANNEL_URL}")
    print()

    ydl_opts = {
        "quiet": False,
        "no_warnings": False,

        # Получаем только метаданные.
        "extract_flat": True,

        # Не скачиваем видео на этом этапе.
        "skip_download": True,

        # Не загружаем плейлист целиком.
        "playlistend": RUTUBE_MAX_VIDEOS,

    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                RUTUBE_CHANNEL_URL,
                download=False,
            )

    except Exception as e:
        print()
        print("❌ Failed to read RUTUBE channel")
        print(f"Error: {e}")
        return []

    if not info:
        print("❌ RUTUBE returned no information")
        return []

    entries = info.get("entries") or []

    videos = []

    for entry in entries:
        if not entry:
            continue

        # extract_flat может вернуть URL + metadata
        video = dict(entry)

        video_id = video.get("id")

        if not video_id:
            continue

        # RUTUBE extractor обычно даёт webpage_url.
        webpage_url = video.get("webpage_url")

        if not webpage_url:
            webpage_url = f"https://rutube.ru/video/{video_id}/"

        video["webpage_url"] = webpage_url

        videos.append(video)

    print(f"Found {len(videos)} channel entries")
    print()

    return videos


def get_video_details(video):
    """
    Получает полную информацию о конкретном RUTUBE видео.

    Это нужно потому, что flat playlist может не содержать
    полной длительности/метаданных.
    """

    url = video.get("webpage_url")

    if not url:
        return None

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                url,
                download=False,
            )

        return info

    except Exception as e:
        print(f"⚠️ Could not read video details: {e}")
        return None


def select_rutube_videos(videos):
    """
    Выбирает подходящие длинные видео.

    Правила:
      - есть ID;
      - есть длительность;
      - длительность >= 20 минут;
      - не Shorts;
      - сортировка по дате публикации, новое сначала.
    """

    print("=" * 70)
    print("🔎 FILTERING RUTUBE VIDEOS")
    print("=" * 70)

    candidates = []

    for index, video in enumerate(videos, start=1):

        url = video.get("webpage_url")

        if not url:
            continue

        # ----------------------------------------------------
        # Если duration уже есть — используем его.
        # Если нет — получаем полную информацию.
        # ----------------------------------------------------

        duration = parse_duration(video.get("duration"))

        full_info = None

        if duration is None:
            print(f"Checking metadata: {url}")

            full_info = get_video_details(video)

            if not full_info:
                continue

            video = full_info
            duration = parse_duration(video.get("duration"))

        if duration is None:
            print("⚠️ Duration unknown — skipping")
            continue

        # ----------------------------------------------------
        # Минимальная длительность
        # ----------------------------------------------------

        if duration < MIN_SOURCE_DURATION:
            print(
                f"⏭ Too short: "
                f"{video.get('title', 'Unknown')} "
                f"({format_duration(duration)})"
            )
            continue

        # ----------------------------------------------------
        # Исключаем Shorts
        # ----------------------------------------------------

        title = (video.get("title") or "").lower()

        if "shorts" in title:
            print(
                f"⏭ Possible Shorts: "
                f"{video.get('title', 'Unknown')}"
            )
            continue

        # ----------------------------------------------------
        # Добавляем кандидата
        # ----------------------------------------------------

        upload_date = video.get("upload_date")

        if upload_date:
            try:
                sort_date = datetime.strptime(
                    upload_date,
                    "%Y%m%d"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                sort_date = datetime.min.replace(
                    tzinfo=timezone.utc
                )
        else:
            timestamp = video.get("timestamp")

            if timestamp:
                try:
                    sort_date = datetime.fromtimestamp(
                        timestamp,
                        tz=timezone.utc,
                    )
                except Exception:
                    sort_date = datetime.min.replace(
                        tzinfo=timezone.utc
                    )
            else:
                sort_date = datetime.min.replace(
                    tzinfo=timezone.utc
                )

        video["_sort_date"] = sort_date

        candidates.append(video)

    # Новые видео первыми
    candidates.sort(
        key=lambda x: x.get("_sort_date"),
        reverse=True,
    )

    print()
    print(f"Suitable videos: {len(candidates)}")
    print()

    for index, video in enumerate(candidates, start=1):
        print_video_info(video, index)
        print()

    return candidates


# ============================================================
# DOWNLOAD
# ============================================================

def download_rutube_video(video):
    """
    Скачивает конкретное RUTUBE видео в:
        input/source.mp4

    Используем yt-dlp + ffmpeg.

    Render pipeline не затрагивается.
    """

    url = video.get("webpage_url")

    if not url:
        print("❌ Video URL is missing")
        return False

    title = video.get("title") or "Unknown"

    print("=" * 70)
    print("⬇️ DOWNLOADING RUTUBE VIDEO")
    print("=" * 70)

    print(f"Title: {title}")
    print(f"URL:   {url}")
    print()

    clean_source_file()

    SOURCE_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Формат:
    #
    # bestvideo до 720p + bestaudio
    # fallback на combined format
    #
    # Для RUTUBE yt-dlp сам получает доступные formats
    # через extractor.
    # --------------------------------------------------------

    ydl_opts = {
        "outtmpl": str(
            SOURCE_VIDEO.with_suffix(".%(ext)s")
        ),

        "format": (
            f"bv*[height<={MAX_VIDEO_HEIGHT}]"
            f"+ba/"
            f"b[height<={MAX_VIDEO_HEIGHT}]"
            f"/b"
        ),

        "merge_output_format": "mp4",

        # ffmpeg
        "ffmpeg_location": "ffmpeg",

        # Сетевые настройки
        "socket_timeout": 30,

        # Повторные попытки
        "retries": 5,
        "fragment_retries": 5,

        # Не оставляем лишние файлы
        "keepvideo": False,

        # Вывод
        "quiet": False,
        "no_warnings": False,

        # Не скачивать плейлист
        "noplaylist": True,
    }

    try:
        start_time = time.time()

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        elapsed = time.time() - start_time

        print()
        print(f"Download finished in {elapsed:.1f}s")

    except Exception as e:
        print()
        print("❌ RUTUBE download failed")
        print(f"Error: {e}")

        clean_source_file()

        return False

    # --------------------------------------------------------
    # Проверяем результат
    # --------------------------------------------------------

    if SOURCE_VIDEO.exists():
        size_mb = SOURCE_VIDEO.stat().st_size / 1024 / 1024

        print()
        print("✅ source.mp4 created")
        print(f"Size: {size_mb:.2f} MB")

        if size_mb < 1:
            print("❌ File is suspiciously small")

            clean_source_file()

            return False

        return True

    # --------------------------------------------------------
    # Иногда yt-dlp/ffmpeg может оставить другой extension.
    # Ищем его.
    # --------------------------------------------------------

    possible_files = list(
        SOURCE_VIDEO.parent.glob("source.*")
    )

    possible_files = [
        path
        for path in possible_files
        if path.is_file()
        and path.name != "source.mp4"
        and path.suffix.lower()
        in {".mkv", ".webm", ".mov", ".mp4", ".m4v"}
    ]

    if possible_files:

        source = max(
            possible_files,
            key=lambda p: p.stat().st_size,
        )

        print(
            f"⚠️ Found downloaded file: "
            f"{source}"
        )

        # Конвертируем в MP4 через ffmpeg.
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source),
                    "-c",
                    "copy",
                    str(SOURCE_VIDEO),
                ],
                check=True,
            )

            source.unlink(missing_ok=True)

        except Exception as e:
            print(
                f"❌ Could not convert source to MP4: {e}"
            )

            return False

        if SOURCE_VIDEO.exists():
            print("✅ Converted to input/source.mp4")
            return True

    print("❌ input/source.mp4 was not created")

    return False


def find_and_download():
    """
    Основная логика:

        RUTUBE channel
              ↓
        список видео
              ↓
        фильтр
              ↓
        новое длинное видео
              ↓
        download
              ↓
        fallback на следующий кандидат
    """

    videos = get_rutube_videos()

    if not videos:
        raise RuntimeError(
            "❌ No videos found on RUTUBE channel."
        )

    candidates = select_rutube_videos(videos)

    if not candidates:
        raise RuntimeError(
            "❌ No suitable long RUTUBE videos found."
        )

    print("=" * 70)
    print("🎯 DOWNLOAD CANDIDATES")
    print("=" * 70)

    # Не больше 5 попыток.
    candidates_to_try = candidates[:5]

    for index, video in enumerate(
        candidates_to_try,
        start=1,
    ):

        print()
        print(
            f"Attempt {index}/"
            f"{len(candidates_to_try)}"
        )

        print_video_info(video)

        if download_rutube_video(video):
            print()
            print("🎉 RUTUBE source ready")
            return True

        if index < len(candidates_to_try):
            print()
            print(
                f"Waiting {RETRY_DELAY}s before next video..."
            )

            time.sleep(RETRY_DELAY)

    raise RuntimeError(
        "❌ None of the RUTUBE candidates "
        "could be downloaded."
    )


# ============================================================
# PROCESS
# ============================================================

def process_video():
    """
    Существующий pipeline проекта.

    ВАЖНО:
    Whisper → Gemini → Render не изменены.
    """

    print("=" * 70)
    print("🎬 PROCESS VIDEO")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(
            "❌ input/source.mp4 not found"
        )

    print()
    print("1️⃣ Validating source video...")

    validate_source_video(SOURCE_VIDEO)

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = get_duration(SOURCE_VIDEO)

    print(
        f"Source duration: "
        f"{format_duration(duration)}"
    )

    # --------------------------------------------------------
    # Extract audio
    # --------------------------------------------------------

    print()
    print("2️⃣ Extracting audio...")

    AUDIO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extract_audio(
        SOURCE_VIDEO,
        AUDIO_FILE,
    )

    # --------------------------------------------------------
    # Whisper
    # --------------------------------------------------------

    print()
    print("3️⃣ Transcribing with Whisper...")

    transcript = transcribe_with_language(
        AUDIO_FILE,
        language="ru",
    )

    if not transcript:
        raise RuntimeError(
            "❌ Whisper returned empty transcript"
        )

    print("✅ Transcript ready")

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    print()
    print("4️⃣ Selecting clips with Gemini...")

    clips = select_clips(
        transcript,
        max_clips=MAX_CLIPS,
    )

    if not clips:
        raise RuntimeError(
            "❌ Gemini did not return clips"
        )

    print()
    print(f"Selected clips: {len(clips)}")

    for index, clip in enumerate(
        clips,
        start=1,
    ):
        print()
        print(f"Clip {index}:")
        print(json.dumps(
            clip,
            ensure_ascii=False,
            indent=2,
        ))

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    print()
    print("5️⃣ Rendering Shorts...")

    CLIPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    render_clips(
        SOURCE_VIDEO,
        clips,
        output_dir=CLIPS_DIR,
    )

    print()
    print("=" * 70)
    print("✅ PROCESS COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("🚀 RUTUBE AI SHORTS")
    print()

    if "--download" in sys.argv:
        print("Mode: DOWNLOAD")
        print()

        find_and_download()

        return

    if "--process" in sys.argv:
        print("Mode: PROCESS")
        print()

        process_video()

        return

    # --------------------------------------------------------
    # Default:
    # download + process
    # --------------------------------------------------------

    print("Mode: DOWNLOAD + PROCESS")
    print()

    find_and_download()

    print()

    process_video()


if __name__ == "__main__":
    main()
