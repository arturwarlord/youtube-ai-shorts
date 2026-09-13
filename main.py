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
from ai.metadata import generate_metadata

from video.clip_renderer import render_clips
from video.subtitles import (
    create_ass_subtitles,
    burn_subtitles,
)


# ============================================================
# CONFIG
# ============================================================

SOURCE_VIDEO = Path("input/source.mp4")
AUDIO_FILE = Path("output/source_audio.wav")
CLIPS_DIR = Path("output/clips")

RUTUBE_CHANNEL_URL = "https://rutube.ru/channel/23968031/"

RUTUBE_MAX_VIDEOS = 20

MIN_SOURCE_DURATION = 20 * 60

MAX_CLIPS = 3

MAX_VIDEO_HEIGHT = 720

DOWNLOAD_TIMEOUT = 25 * 60

RETRY_DELAY = 3


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


def clean_source_file():
    """
    Удаляет старый source.mp4.
    """

    SOURCE_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SOURCE_VIDEO.exists():
        print("🗑 Removing old source.mp4")
        SOURCE_VIDEO.unlink()


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
    """
    Выбирает подходящие длинные видео.
    """

    print("=" * 70)
    print("🔎 FILTERING RUTUBE VIDEOS")
    print("=" * 70)

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

    clean_source_file()

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

    print("=" * 70)
    print("🎬 PROCESS VIDEO")
    print("=" * 70)

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

    clips = select_clips(
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
    # 6 + 7. Subtitles + Metadata
    # --------------------------------------------------------

    print()

    print(
        "6️⃣ Adding subtitles..."
    )

    print()

    print(
        "7️⃣ Generating metadata..."
    )

    final_videos = []

    metadata_files = []

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
        # Сохраняем JSON
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

    print(
        "=" * 70
    )

    print(
        "✅ PROCESS COMPLETE"
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
