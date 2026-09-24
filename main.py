import os
import sys
import json
import time
import re
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
from ai.metadata import generate_metadata
from video.thumbnails import create_thumbnail

from video.clip_renderer import render_clips
from video.subtitles import (
    create_ass_subtitles,
    burn_subtitles,
)

from youtube.upload import upload_video


# ============================================================
# CONFIG
# ============================================================

SOURCE_VIDEO = Path("input/source.mp4")
AUDIO_FILE = Path("output/source_audio.wav")
CLIPS_DIR = Path("output/clips")

RUTUBE_CHANNEL_URL = "https://rutube.ru/channel/23968031/"

# Сканируем больше записей канала, потому что последние 50 видео
# могут быть в основном короткими Shorts. extract_flat не скачивает
# сами видео — только метаданные, поэтому увеличение лимита безопасно.
RUTUBE_MAX_VIDEOS = 300

MIN_SOURCE_DURATION = 20 * 60

# 2 Shorts за один запуск.
# Workflow запускается 3 раза в день = 6 Shorts / день.
MAX_CLIPS = 2

MAX_VIDEO_HEIGHT = 720

DOWNLOAD_TIMEOUT = 25 * 60

RETRY_DELAY = 3

# ============================================================
# GEMINI RETRY
# ============================================================

# Максимальное количество повторных попыток Gemini
# при ошибке 429 RESOURCE_EXHAUSTED.
GEMINI_MAX_RETRIES = 10

# Небольшой запас к времени, которое сообщает Gemini.
# Например:
# Please retry in 33.5s
# реально ждём 35.0s.
GEMINI_RETRY_BUFFER = 1.5

# Если Gemini вернул 429, но время ожидания
# в тексте ошибки определить не удалось.
GEMINI_DEFAULT_RETRY_DELAY = 60

# ============================================================
# PROCESSING HISTORY
# ============================================================

# This file is intentionally stored in the repository so GitHub
# Actions runs remember which RUTUBE source videos were already
# processed and published.
HISTORY_FILE = Path("data/processed_videos.json")
SOURCE_INFO_FILE = Path("input/source.json")

# Filled during --download and recovered during --process.
SELECTED_SOURCE_VIDEO_INFO = {}


def load_processing_history():
    """Load persistent RUTUBE processing history."""

    if not HISTORY_FILE.exists():
        return {"videos": {}}

    try:
        data = json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )
    except Exception as error:
        print(
            f"⚠️ Could not read {HISTORY_FILE}: {error}"
        )
        return {"videos": {}}

    if not isinstance(data, dict):
        return {"videos": {}}

    videos = data.get("videos")

    if not isinstance(videos, dict):
        data["videos"] = {}

    return data


def save_processing_history(history):
    """Atomically save processing history."""

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    temp_file = HISTORY_FILE.with_suffix(".tmp")

    temp_file.write_text(
        json.dumps(
            history,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temp_file.replace(HISTORY_FILE)


def save_source_info(video):
    """Persist the selected source between --download and --process."""

    SOURCE_INFO_FILE.parent.mkdir(parents=True, exist_ok=True)

    SOURCE_INFO_FILE.write_text(
        json.dumps(video, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_source_info():
    """Recover selected source metadata for a separate --process run."""

    if not SOURCE_INFO_FILE.exists():
        return {}

    try:
        data = json.loads(
            SOURCE_INFO_FILE.read_text(encoding="utf-8")
        )
    except Exception as error:
        print(
            f"⚠️ Could not read {SOURCE_INFO_FILE}: {error}"
        )
        return {}

    return data if isinstance(data, dict) else {}


def mark_video_processed(video, clips, uploaded_videos):
    """Record a source only after all selected Shorts were uploaded."""

    video_id = str(video.get("id") or "").strip()

    if not video_id:
        print("⚠️ Cannot persist history: source video ID is missing.")
        return False

    history = load_processing_history()

    clip_records = []

    for index, clip in enumerate(clips):
        youtube_id = (
            uploaded_videos[index]
            if index < len(uploaded_videos)
            else None
        )

        clip_records.append({
            "start": float(clip["start"]),
            "end": float(clip["end"]),
            "youtube_video_id": youtube_id,
        })

    history["videos"][video_id] = {
        "id": video_id,
        "title": video.get("title") or "",
        "webpage_url": video.get("webpage_url") or "",
        "upload_date": video.get("upload_date") or "",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "clips": clip_records,
    }

    save_processing_history(history)

    print()
    print("💾 Processing history updated")
    print(f"   Source ID: {video_id}")
    print(f"   History:   {HISTORY_FILE}")

    return True


def persist_history_to_git():
    """Commit/push processing history when running inside GitHub Actions."""

    if not os.environ.get("GITHUB_ACTIONS"):
        print("ℹ️ Local run: processing history was saved locally.")
        return True

    try:
        subprocess.run(
            ["git", "config", "user.name", "github-actions[bot]"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        subprocess.run(
            ["git", "add", str(HISTORY_FILE)],
            check=True,
        )

        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", str(HISTORY_FILE)],
        )

        if diff.returncode == 0:
            print("ℹ️ Processing history has no new Git changes.")
            return True

        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "chore: update processed video history",
            ],
            check=True,
        )

        subprocess.run(
            ["git", "push"],
            check=True,
        )

        print("✅ Processing history pushed to GitHub")
        return True

    except subprocess.CalledProcessError as error:
        print()
        print("⚠️ Could not persist processing history to GitHub.")
        print(
            "Make sure the workflow has "
            "permissions: contents: write."
        )
        print(f"Git error: {error}")
        return False


# ============================================================
# HELPERS
# ============================================================

def parse_duration(value):
    """
    Преобразует длительность в секунды.

    Поддерживает:
      - int / float
      - HH:MM:SS
      - MM:SS
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


def clean_source_file(remove_source_info=False):
    """Удаляет старый source.mp4 и, при необходимости, его metadata."""

    SOURCE_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SOURCE_VIDEO.exists():
        print("🗑 Removing old source.mp4")
        SOURCE_VIDEO.unlink()

    if remove_source_info and SOURCE_INFO_FILE.exists():
        print("🗑 Removing old source.json")
        SOURCE_INFO_FILE.unlink()


def print_video_info(video, index=None):
    title = video.get("title") or "Unknown title"
    video_id = video.get("id") or "unknown"

    duration = parse_duration(
        video.get("duration")
    )

    url = (
        video.get("webpage_url")
        or video.get("url")
        or ""
    )

    prefix = (
        f"[{index}] "
        if index is not None
        else ""
    )

    print(
        f"{prefix}{title}\n"
        f"    ID: {video_id}\n"
        f"    Duration: {format_duration(duration)}\n"
        f"    URL: {url}"
    )


# ============================================================
# GEMINI RETRY HELPERS
# ============================================================

def is_gemini_rate_limit_error(error):
    """
    Проверяет, является ли ошибка Gemini
    ошибкой 429 RESOURCE_EXHAUSTED.

    Мы специально не анализируем/не меняем сам
    Gemini selection. Только определяем,
    можно ли повторить запрос.
    """

    error_text = str(error).lower()

    if "429" in error_text:
        return True

    if "resource_exhausted" in error_text:
        return True

    if "quota exceeded" in error_text:
        return True

    if "generate_content_free_tier" in error_text:
        return True

    return False


def get_gemini_retry_delay(error):
    """
    Достаёт время ожидания из сообщения Gemini.

    Например Gemini возвращает:

        Please retry in 33.533170356s.

    Функция вернёт:

        33.533170356
    """

    error_text = str(error)

    patterns = [
        r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retryDelay[\"']?\s*[:=]\s*[\"']?([0-9]+(?:\.[0-9]+)?)s",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            error_text,
            flags=re.IGNORECASE,
        )

        if match:

            try:
                return float(
                    match.group(1)
                )

            except (
                ValueError,
                TypeError,
            ):
                pass

    return None


def select_clips_with_retry(
    transcript_words,
    max_clips,
):
    """
    Вызывает оригинальный select_clips()
    без изменения его логики.

    Единственное отличие:
    если Gemini возвращает 429 RESOURCE_EXHAUSTED,
    ждём ровно столько, сколько рекомендует Gemini,
    и повторяем тот же запрос.

    Пример:

        429
        Please retry in 33.5s

        ↓

        sleep 35.0s

        ↓

        select_clips() повторно
    """

    attempt = 0

    while True:

        attempt += 1

        print()

        if attempt == 1:

            print(
                "🤖 Calling Gemini clip selector..."
            )

        else:

            print(
                f"🤖 Retrying Gemini clip selector "
                f"(attempt {attempt}/"
                f"{GEMINI_MAX_RETRIES + 1})..."
            )

        try:

            # ====================================================
            # ВАЖНО:
            # Здесь вызывается оригинальный select_clips()
            # без каких-либо изменений.
            # ====================================================

            return select_clips(
                transcript_words,
                max_clips=max_clips,
            )

        except Exception as error:

            if not is_gemini_rate_limit_error(
                error
            ):

                # Это НЕ 429.
                # Оставляем поведение как раньше:
                # ошибка сразу выходит наружу.
                raise

            # ----------------------------------------------------
            # 429 RESOURCE_EXHAUSTED
            # ----------------------------------------------------

            if (
                attempt
                > GEMINI_MAX_RETRIES
            ):

                print()

                print(
                    "❌ Gemini rate limit "
                    "retry limit exceeded."
                )

                print(
                    f"Maximum retries: "
                    f"{GEMINI_MAX_RETRIES}"
                )

                raise

            retry_delay = (
                get_gemini_retry_delay(
                    error
                )
            )

            if retry_delay is None:

                retry_delay = (
                    GEMINI_DEFAULT_RETRY_DELAY
                )

                print()

                print(
                    "⚠️ Gemini returned 429, "
                    "but retry time could not "
                    "be detected."
                )

                print(
                    f"Using default wait: "
                    f"{retry_delay:.1f}s"
                )

            else:

                print()

                print(
                    "⚠️ Gemini rate limit "
                    "reached."
                )

                print(
                    f"⏳ Gemini requested retry "
                    f"in {retry_delay:.3f}s"
                )

            # Добавляем небольшой запас,
            # чтобы не повторить запрос ровно
            # в момент окончания лимита.
            wait_time = (
                retry_delay
                + GEMINI_RETRY_BUFFER
            )

            print(
                f"⏳ Waiting "
                f"{wait_time:.1f}s "
                f"before retry..."
            )

            # Показываем обратный отсчёт
            # в Actions, чтобы было понятно,
            # что процесс не завис.
            remaining = wait_time

            while remaining > 0:

                sleep_for = min(
                    10,
                    remaining,
                )

                print(
                    f"   ⏱ {remaining:.1f}s remaining..."
                )

                time.sleep(
                    sleep_for
                )

                remaining -= sleep_for

            print()

            print(
                "🔄 Retrying Gemini request..."
            )


# ============================================================
# RUTUBE
# ============================================================

def get_rutube_videos():
    """
    Получает список видео с RUTUBE-канала.
    """

    print("=" * 70)
    print("📺 RUTUBE")
    print("=" * 70)

    print(
        f"Channel: {RUTUBE_CHANNEL_URL}"
    )

    print()

    ydl_opts = {
        "quiet": False,
        "no_warnings": False,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": RUTUBE_MAX_VIDEOS,
    }

    try:

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            info = ydl.extract_info(
                RUTUBE_CHANNEL_URL,
                download=False,
            )

    except Exception as e:

        print()
        print(
            "❌ Failed to read RUTUBE channel"
        )

        print(
            f"Error: {e}"
        )

        return []

    if not info:

        print(
            "❌ RUTUBE returned no information"
        )

        return []

    entries = (
        info.get("entries")
        or []
    )

    videos = []

    for entry in entries:

        if not entry:
            continue

        video = dict(entry)

        video_id = video.get("id")

        if not video_id:
            continue

        webpage_url = (
            video.get("webpage_url")
        )

        if not webpage_url:

            webpage_url = (
                f"https://rutube.ru/video/"
                f"{video_id}/"
            )

        video["webpage_url"] = webpage_url

        videos.append(video)

    print(
        f"Found {len(videos)} channel entries"
    )

    print()

    return videos


def get_video_details(video):
    """
    Получает полную информацию
    о конкретном RUTUBE-видео.
    """

    url = video.get(
        "webpage_url"
    )

    if not url:
        return None

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=False,
            )

        return info

    except Exception as e:

        print(
            f"⚠️ Could not read "
            f"video details: {e}"
        )

        return None


def select_rutube_videos(videos):
    """Выбирает подходящие длинные и ещё не обработанные видео."""

    print("=" * 70)
    print("🔎 FILTERING RUTUBE VIDEOS")
    print("=" * 70)

    history = load_processing_history()
    processed_ids = {
        str(video_id)
        for video_id in history.get("videos", {}).keys()
    }

    print(
        f"Previously processed sources: {len(processed_ids)}"
    )

    candidates = []

    for index, video in enumerate(
        videos,
        start=1,
    ):

        url = video.get(
            "webpage_url"
        )

        if not url:
            continue

        video_id = str(video.get("id") or "").strip()

        if not video_id:
            continue

        if video_id in processed_ids:
            print(
                f"⏭ Already processed: "
                f"{video.get('title', 'Unknown')} "
                f"(ID: {video_id})"
            )
            continue

        duration = parse_duration(
            video.get("duration")
        )

        if duration is None:

            print(
                f"Checking metadata: {url}"
            )

            full_info = (
                get_video_details(video)
            )

            if not full_info:
                continue

            video = full_info

            duration = parse_duration(
                video.get("duration")
            )

        if duration is None:

            print(
                "⚠️ Duration unknown — skipping"
            )

            continue

        if duration < MIN_SOURCE_DURATION:

            print(
                f"⏭ Too short: "
                f"{video.get('title', 'Unknown')} "
                f"({format_duration(duration)})"
            )

            continue

        title = (
            video.get("title")
            or ""
        ).lower()

        if "shorts" in title:

            print(
                f"⏭ Possible Shorts: "
                f"{video.get('title', 'Unknown')}"
            )

            continue

        upload_date = video.get(
            "upload_date"
        )

        if upload_date:

            try:

                sort_date = (
                    datetime.strptime(
                        upload_date,
                        "%Y%m%d",
                    ).replace(
                        tzinfo=timezone.utc
                    )
                )

            except Exception:

                sort_date = (
                    datetime.min.replace(
                        tzinfo=timezone.utc
                    )
                )

        else:

            timestamp = video.get(
                "timestamp"
            )

            if timestamp:

                try:

                    sort_date = (
                        datetime.fromtimestamp(
                            timestamp,
                            tz=timezone.utc,
                        )
                    )

                except Exception:

                    sort_date = (
                        datetime.min.replace(
                            tzinfo=timezone.utc
                        )
                    )

            else:

                sort_date = (
                    datetime.min.replace(
                        tzinfo=timezone.utc
                    )
                )

        video["_sort_date"] = sort_date

        candidates.append(video)

    candidates.sort(
        key=lambda x: x.get(
            "_sort_date"
        ),
        reverse=True,
    )

    print()

    print(
        f"Suitable videos: "
        f"{len(candidates)}"
    )

    print()

    for index, video in enumerate(
        candidates,
        start=1,
    ):

        print_video_info(
            video,
            index,
        )

        print()

    return candidates


# ============================================================
# DOWNLOAD
# ============================================================

def download_rutube_video(video):
    """
    Скачивает RUTUBE-видео в:

        input/source.mp4
    """

    url = video.get(
        "webpage_url"
    )

    if not url:

        print(
            "❌ Video URL is missing"
        )

        return False

    title = (
        video.get("title")
        or "Unknown"
    )

    print("=" * 70)
    print("⬇️ DOWNLOADING RUTUBE VIDEO")
    print("=" * 70)

    print(
        f"Title: {title}"
    )

    print(
        f"URL:   {url}"
    )

    print()

    clean_source_file(remove_source_info=True)

    SOURCE_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ydl_opts = {
        "outtmpl": str(
            SOURCE_VIDEO.with_suffix(
                ".%(ext)s"
            )
        ),

        "format": (
            f"bv*[height<={MAX_VIDEO_HEIGHT}]"
            f"+ba/"
            f"b[height<={MAX_VIDEO_HEIGHT}]"
            f"/b"
        ),

        "merge_output_format": "mp4",

        "ffmpeg_location": "ffmpeg",

        "socket_timeout": 30,

        "retries": 5,

        "fragment_retries": 5,

        "keepvideo": False,

        "quiet": False,

        "no_warnings": False,

        "noplaylist": True,
    }

    try:

        start_time = time.time()

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            ydl.download([url])

        elapsed = (
            time.time()
            - start_time
        )

        print()

        print(
            f"Download finished in "
            f"{elapsed:.1f}s"
        )

    except Exception as e:

        print()

        print(
            "❌ RUTUBE download failed"
        )

        print(
            f"Error: {e}"
        )

        clean_source_file()

        return False

    if SOURCE_VIDEO.exists():

        size_mb = (
            SOURCE_VIDEO.stat().st_size
            / 1024
            / 1024
        )

        print()

        print(
            "✅ source.mp4 created"
        )

        print(
            f"Size: {size_mb:.2f} MB"
        )

        if size_mb < 1:

            print(
                "❌ File is suspiciously small"
            )

            clean_source_file()

            return False

        save_source_info(video)
        print(f"💾 Source metadata saved: {SOURCE_INFO_FILE}")

        return True

    possible_files = list(
        SOURCE_VIDEO.parent.glob(
            "source.*"
        )
    )

    possible_files = [
        path
        for path in possible_files
        if path.is_file()
        and path.name != "source.mp4"
        and path.suffix.lower()
        in {
            ".mkv",
            ".webm",
            ".mov",
            ".mp4",
            ".m4v",
        }
    ]

    if possible_files:

        source = max(
            possible_files,
            key=lambda p:
            p.stat().st_size,
        )

        print(
            f"⚠️ Found downloaded file: "
            f"{source}"
        )

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

            source.unlink(
                missing_ok=True
            )

        except Exception as e:

            print(
                f"❌ Could not convert "
                f"source to MP4: {e}"
            )

            return False

        if SOURCE_VIDEO.exists():

            print(
                "✅ Converted to "
                "input/source.mp4"
            )

            save_source_info(video)
            print(f"💾 Source metadata saved: {SOURCE_INFO_FILE}")

            return True

    print(
        "❌ input/source.mp4 "
        "was not created"
    )

    return False


def find_and_download():

    videos = get_rutube_videos()

    if not videos:

        raise RuntimeError(
            "❌ No videos found "
            "on RUTUBE channel."
        )

    candidates = (
        select_rutube_videos(
            videos
        )
    )

    if not candidates:

        raise RuntimeError(
            "❌ No suitable long "
            "RUTUBE videos found."
        )

    print("=" * 70)
    print("🎯 DOWNLOAD CANDIDATES")
    print("=" * 70)

    candidates_to_try = candidates[:10]

    if not candidates_to_try:
        raise RuntimeError(
            "❌ All recent RUTUBE sources were already processed."
        )

    for index, video in enumerate(
        candidates_to_try,
        start=1,
    ):

        print()

        print(
            f"Attempt {index}/"
            f"{len(candidates_to_try)}"
        )

        print_video_info(
            video
        )

        if download_rutube_video(
            video
        ):

            print()

            print(
                "🎉 RUTUBE source ready"
            )

            return True

        if (
            index
            < len(candidates_to_try)
        ):

            print()

            print(
                f"Waiting {RETRY_DELAY}s "
                f"before next video..."
            )

            time.sleep(
                RETRY_DELAY
            )

    raise RuntimeError(
        "❌ None of the RUTUBE "
        "candidates could be downloaded."
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video():

    global SELECTED_SOURCE_VIDEO_INFO

    print("=" * 70)
    print("🎬 PROCESS VIDEO")
    print("=" * 70)

    if not SELECTED_SOURCE_VIDEO_INFO:
        SELECTED_SOURCE_VIDEO_INFO = load_source_info()

    if SELECTED_SOURCE_VIDEO_INFO:
        print()
        print(
            "📺 Source: "
            f"{SELECTED_SOURCE_VIDEO_INFO.get('title', 'Unknown')}"
        )
        print(
            "🆔 Source ID: "
            f"{SELECTED_SOURCE_VIDEO_INFO.get('id', 'unknown')}"
        )

    # --------------------------------------------------------
    # 1. Validate
    # --------------------------------------------------------

    if not SOURCE_VIDEO.exists():

        raise FileNotFoundError(
            "❌ input/source.mp4 not found"
        )

    print()

    print(
        "1️⃣ Validating source video..."
    )

    validate_source_video(
        SOURCE_VIDEO
    )

    duration = get_duration(
        SOURCE_VIDEO
    )

    print(
        f"Source duration: "
        f"{format_duration(duration)}"
    )

    # --------------------------------------------------------
    # 2. Audio
    # --------------------------------------------------------

    print()

    print(
        "2️⃣ Extracting audio..."
    )

    AUDIO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extract_audio(
        SOURCE_VIDEO,
        AUDIO_FILE,
    )

    # --------------------------------------------------------
    # 3. Whisper
    # --------------------------------------------------------

    print()

    print(
        "3️⃣ Transcribing with Whisper..."
    )

    transcript = (
        transcribe_with_language(
            AUDIO_FILE,
        )
    )

    if not transcript:

        raise RuntimeError(
            "❌ Whisper returned "
            "empty transcript"
        )

    if not transcript.get(
        "words"
    ):

        raise RuntimeError(
            "❌ Whisper returned "
            "no word timestamps"
        )

    print(
        "✅ Transcript ready"
    )

    print(
        f"📝 Words: "
        f"{len(transcript['words'])}"
    )

    # --------------------------------------------------------
    # 4. Gemini — select clips
    # --------------------------------------------------------

    print()

    print(
        "4️⃣ Selecting clips with Gemini..."
    )

    # ========================================================
    # ВАЖНО:
    #
    # Сам select_clips() НЕ ИЗМЕНЁН.
    #
    # Добавлен только внешний retry для 429.
    # Если Gemini сообщает:
    #
    # Please retry in 33.533170356s
    #
    # скрипт ждёт это время + небольшой запас
    # и повторяет ТОТ ЖЕ запрос.
    # ========================================================

    clips = select_clips_with_retry(
        transcript["words"],
        max_clips=MAX_CLIPS,
    )

    if not clips:

        raise RuntimeError(
            "❌ Gemini did not "
            "return clips"
        )

    print()

    print(
        f"Selected clips: "
        f"{len(clips)}"
    )

    for index, clip in enumerate(
        clips,
        start=1,
    ):

        print()

        print(
            f"Clip {index}:"
        )

        print(
            json.dumps(
                clip,
                ensure_ascii=False,
                indent=2,
            )
        )

    # --------------------------------------------------------
    # 5. Render
    # --------------------------------------------------------

    print()

    print(
        "5️⃣ Rendering Shorts..."
    )

    CLIPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered = render_clips(
        SOURCE_VIDEO,
        clips,
        output_dir=CLIPS_DIR,
    )

    if not rendered:

        raise RuntimeError(
            "❌ No clips were rendered"
        )

    if len(rendered) != len(
        clips
    ):

        raise RuntimeError(
            "❌ Number of rendered "
            "clips does not match "
            "selected clips"
        )

    print()

    print(
        f"✅ Rendered clips: "
        f"{len(rendered)}"
    )

    # --------------------------------------------------------
    # 6 + 7 + 8. Subtitles + Metadata + Thumbnails
    # --------------------------------------------------------

    print()

    print(
        "6️⃣ Adding subtitles..."
    )

    print()

    print(
        "7️⃣ Generating metadata..."
    )

    print()

    print(
        "8️⃣ Generating thumbnails..."
    )

    final_videos = []

    metadata_files = []

    thumbnail_files = []

    uploaded_videos = []

    for index, (
        rendered_path,
        clip,
    ) in enumerate(
        zip(
            rendered,
            clips,
        ),
        start=1,
    ):

        subtitle_path = (
            CLIPS_DIR
            / f"clip_{index:02d}.ass"
        )

        final_path = (
            CLIPS_DIR
            / f"clip_{index:02d}_final.mp4"
        )

        metadata_path = (
            CLIPS_DIR
            / f"clip_{index:02d}_metadata.json"
        )

        thumbnail_path = (
            CLIPS_DIR
            / f"clip_{index:02d}_thumbnail.jpg"
        )

        print()

        print(
            "=" * 60
        )

        print(
            f"🎬 Processing clip "
            f"{index}/{len(clips)}"
        )

        print(
            f"Video: {rendered_path}"
        )

        print(
            f"Time: "
            f"{float(clip['start']):.2f} → "
            f"{float(clip['end']):.2f}"
        )

        print(
            f"Subtitle: "
            f"{subtitle_path}"
        )

        print(
            f"Final: "
            f"{final_path}"
        )

        print(
            f"Metadata: "
            f"{metadata_path}"
        )

        print(
            f"Thumbnail: "
            f"{thumbnail_path}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # SUBTITLES
        # ----------------------------------------------------

        create_ass_subtitles(
            transcript["words"],
            float(
                clip["start"]
            ),
            float(
                clip["end"]
            ),
            subtitle_path,
        )

        burn_subtitles(
            rendered_path,
            subtitle_path,
            final_path,
        )

        if not final_path.exists():

            raise RuntimeError(
                "❌ Final subtitle "
                "video was not created: "
                f"{final_path}"
            )

        final_videos.append(
            str(final_path)
        )

        print()

        print(
            "✅ Subtitle video created"
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        print()

        print(
            f"🤖 Generating metadata "
            f"for clip {index}..."
        )

        metadata = generate_metadata(
            clip
        )

        if not metadata:

            raise RuntimeError(
                "❌ Metadata generation "
                f"failed for clip {index}"
            )

        # ----------------------------------------------------
        # SAVE METADATA JSON
        # ----------------------------------------------------

        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        metadata_files.append(
            str(metadata_path)
        )

        print()

        print(
            "✅ Metadata created"
        )

        print(
            f"Title: "
            f"{metadata.get('title', '')}"
        )

        print(
            "Description:"
        )

        print(
            metadata.get(
                "description",
                "",
            )
        )

        print(
            "Hashtags: "
            + " ".join(
                metadata.get(
                    "hashtags",
                    [],
                )
            )
        )

        print(
            "Tags: "
            + ", ".join(
                metadata.get(
                    "tags",
                    [],
                )
            )
        )

        print(
            f"📁 {metadata_path}"
        )

        # ----------------------------------------------------
        # THUMBNAIL
        # ----------------------------------------------------

        print()

        print(
            f"🖼️ Generating thumbnail "
            f"for clip {index}..."
        )

        create_thumbnail(
            final_path,
            metadata_path,
            thumbnail_path,
        )

        if not thumbnail_path.exists():

            raise RuntimeError(
                "❌ Thumbnail was not created: "
                f"{thumbnail_path}"
            )

        thumbnail_files.append(
            str(thumbnail_path)
        )

        print(
            "✅ Thumbnail created"
        )

        print(
            f"📁 {thumbnail_path}"
        )

        # ----------------------------------------------------
        # YOUTUBE UPLOAD
        # ----------------------------------------------------

        print()

        print(
            "=" * 60
        )

        print(
            f"📤 UPLOADING CLIP "
            f"{index}/{len(clips)} TO YOUTUBE"
        )

        print(
            "=" * 60
        )

        title = str(
            metadata.get(
                "title",
                "",
            )
        ).strip()

        description = str(
            metadata.get(
                "description",
                "",
            )
        ).strip()

        hashtags = metadata.get(
            "hashtags",
            [],
        )

        tags = metadata.get(
            "tags",
            [],
        )

        if not title:

            raise RuntimeError(
                f"❌ Empty YouTube title "
                f"for clip {index}"
            )

        if not description:

            raise RuntimeError(
                f"❌ Empty YouTube description "
                f"for clip {index}"
            )

        if not isinstance(
            hashtags,
            list,
        ):

            hashtags = []

        if not isinstance(
            tags,
            list,
        ):

            tags = []

        # upload_video() ожидает hashtags
        # как готовую строку.
        hashtags_text = " ".join(
            str(item).strip()
            for item in hashtags
            if str(item).strip()
        )

        # tags должны передаваться списком.
        clean_tags = []

        for tag in tags:

            tag = str(tag).strip()

            if not tag:
                continue

            if tag.startswith("#"):
                tag = tag[1:]

            if tag not in clean_tags:
                clean_tags.append(tag)

        youtube_video_id = upload_video(
            video_path=str(final_path),
            thumbnail_path=str(thumbnail_path),
            title=title,
            description=description,
            hashtags=hashtags_text,
            tags=clean_tags,
        )

        if not youtube_video_id:

            raise RuntimeError(
                f"❌ YouTube upload failed "
                f"for clip {index}"
            )

        uploaded_videos.append(
            youtube_video_id
        )

        print()

        print(
            f"✅ Clip {index} published "
            f"to YouTube"
        )

        print(
            f"🎬 Video ID: "
            f"{youtube_video_id}"
        )

        print(
            "🔗 "
            f"https://www.youtube.com/watch?v="
            f"{youtube_video_id}"
        )

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "🎉 SHORTS PIPELINE READY"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # FINAL VIDEOS
    # --------------------------------------------------------

    print(
        "🎬 Final videos:"
    )

    for index, video_path in enumerate(
        final_videos,
        start=1,
    ):

        path = Path(
            video_path
        )

        if not path.exists():
            continue

        size_mb = (
            path.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"  ✅ Clip {index}: "
            f"{path}"
        )

        print(
            f"     Size: "
            f"{size_mb:.2f} MB"
        )

    print()

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    print(
        "📝 Metadata files:"
    )

    for index, metadata_path in enumerate(
        metadata_files,
        start=1,
    ):

        print(
            f"  ✅ Clip {index}: "
            f"{metadata_path}"
        )

    print()

    # --------------------------------------------------------
    # THUMBNAILS
    # --------------------------------------------------------

    print(
        "🖼️ Thumbnail files:"
    )

    for index, thumbnail_path in enumerate(
        thumbnail_files,
        start=1,
    ):

        path = Path(
            thumbnail_path
        )

        if not path.exists():
            continue

        size_mb = (
            path.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"  ✅ Clip {index}: "
            f"{path}"
        )

        print(
            f"     Size: "
            f"{size_mb:.2f} MB"
        )

    print()

    # --------------------------------------------------------
    # YOUTUBE
    # --------------------------------------------------------

    print(
        "📺 YouTube uploads:"
    )

    for index, video_id in enumerate(
        uploaded_videos,
        start=1,
    ):

        print(
            f"  ✅ Clip {index}: "
            f"{video_id}"
        )

        print(
            f"     https://www.youtube.com/watch?v="
            f"{video_id}"
        )

    print()

    # --------------------------------------------------------
    # FINAL COUNTS
    # --------------------------------------------------------

    print(
        "📊 Pipeline summary:"
    )

    print(
        f"   Clips selected:     {len(clips)}"
    )

    print(
        f"   Videos rendered:    {len(final_videos)}"
    )

    print(
        f"   Metadata created:   {len(metadata_files)}"
    )

    print(
        f"   Thumbnails created: {len(thumbnail_files)}"
    )

    print(
        f"   YouTube uploads:    {len(uploaded_videos)}"
    )

    print()

    if len(final_videos) != len(clips):

        raise RuntimeError(
            "❌ Final video count does not "
            "match selected clip count"
        )

    if len(metadata_files) != len(clips):

        raise RuntimeError(
            "❌ Metadata count does not "
            "match selected clip count"
        )

    if len(thumbnail_files) != len(clips):

        raise RuntimeError(
            "❌ Thumbnail count does not "
            "match selected clip count"
        )

    if len(uploaded_videos) != len(clips):

        raise RuntimeError(
            "❌ YouTube upload count does not "
            "match selected clip count"
        )

    # --------------------------------------------------------
    # PERSIST SOURCE HISTORY
    # --------------------------------------------------------

    if SELECTED_SOURCE_VIDEO_INFO:
        history_saved = mark_video_processed(
            SELECTED_SOURCE_VIDEO_INFO,
            clips,
            uploaded_videos,
        )

        if history_saved:
            persist_history_to_git()
    else:
        print()
        print(
            "⚠️ Source metadata is unavailable; "
            "processed source was not added to history."
        )

    print(
        "=" * 70
    )

    print(
        "✅ PROCESS COMPLETE"
    )

    print(
        "📺 All selected Shorts uploaded to YouTube"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "🚀 RUTUBE AI SHORTS"
    )

    print()

    if "--download" in sys.argv:

        print(
            "Mode: DOWNLOAD"
        )

        print()

        find_and_download()

        return

    if "--process" in sys.argv:

        print(
            "Mode: PROCESS"
        )

        print()

        process_video()

        return

    print(
        "Mode: DOWNLOAD + PROCESS"
    )

    print()

    find_and_download()

    print()

    process_video()


if __name__ == "__main__":
    main()
