import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests


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

MAX_CLIPS = 3


# ------------------------------------------------------------
# YouTube
# ------------------------------------------------------------

YOUTUBE_API_URL = (
    "https://www.googleapis.com/youtube/v3"
)

DEFAULT_SEARCH_QUERY = (
    "viral podcast interview"
)

SEARCH_RESULTS = 10

SEARCH_DAYS = 14


# ------------------------------------------------------------
# yt-dlp
# ------------------------------------------------------------

MAX_VIDEO_HEIGHT = 720

DOWNLOAD_TIMEOUT = 25 * 60


# ============================================================
# HELPERS
# ============================================================

def print_header():

    print("")
    print("=" * 70)
    print("🚀 AI LONG VIDEO → YOUTUBE SHORTS")
    print("=" * 70)
    print("")


# ============================================================
# YOUTUBE API
# ============================================================

def search_youtube():

    api_key = os.getenv(
        "YOUTUBE_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "YOUTUBE_API_KEY is not set"
        )


    query = os.getenv(
        "SEARCH_QUERY",
        ""
    ).strip()


    if not query:

        query = DEFAULT_SEARCH_QUERY


    print("")
    print("=" * 70)
    print("🔎 YOUTUBE SEARCH")
    print("=" * 70)

    print(
        f"Query: {query}"
    )


    published_after = (
        datetime.now(timezone.utc)
        - timedelta(days=SEARCH_DAYS)
    ).isoformat().replace(
        "+00:00",
        "Z"
    )


    params = {

        "part": "snippet",

        "q": query,

        "type": "video",

        "maxResults": SEARCH_RESULTS,

        "order": "viewCount",

        "publishedAfter": published_after,

        "videoDefinition": "high",

        "safeSearch": "moderate",

        "key": api_key,
    }


    response = requests.get(

        f"{YOUTUBE_API_URL}/search",

        params=params,

        timeout=30,
    )


    if response.status_code != 200:

        raise RuntimeError(

            "YouTube API error: "

            f"{response.status_code}\n"

            f"{response.text[:2000]}"
        )


    data = response.json()


    items = data.get(
        "items",
        []
    )


    if not items:

        raise RuntimeError(
            "YouTube API returned no videos"
        )


    candidates = []


    for item in items:

        video_id = (
            item
            .get("id", {})
            .get("videoId")
        )


        if not video_id:
            continue


        snippet = item.get(
            "snippet",
            {}
        )


        title = snippet.get(
            "title",
            ""
        )


        channel = snippet.get(
            "channelTitle",
            ""
        )


        published = snippet.get(
            "publishedAt",
            ""
        )


        url = (
            "https://www.youtube.com/watch?v="
            + video_id
        )


        candidates.append({

            "video_id": video_id,

            "title": title,

            "channel": channel,

            "published": published,

            "url": url,

        })


    if not candidates:

        raise RuntimeError(
            "No valid YouTube video IDs found"
        )


    # --------------------------------------------------------
    # GET VIDEO DETAILS
    # --------------------------------------------------------

    video_ids = [
        item["video_id"]
        for item in candidates
    ]


    details = get_video_details(
        video_ids,
        api_key,
    )


    result = []


    for item in candidates:

        detail = details.get(
            item["video_id"]
        )


        if not detail:
            continue


        duration = detail[
            "duration"
        ]


        views = detail[
            "views"
        ]


        # We want long-form videos.
        #
        # Minimum: 4 minutes.
        #
        # This avoids Shorts and very short clips.

        if duration < 240:

            continue


        item["duration"] = duration

        item["views"] = views


        result.append(
            item
        )


    # If filtering removed everything,
    # use the original candidates.

    if not result:

        result = candidates


    # Highest views first.

    result.sort(
        key=lambda item: item.get(
            "views",
            0
        ),
        reverse=True,
    )


    print("")

    print(
        f"Found {len(result)} candidates"
    )


    for index, item in enumerate(
        result,
        start=1
    ):

        duration = item.get(
            "duration",
            0
        )


        views = item.get(
            "views",
            0
        )


        print(
            f"{index}. "
            f"{item['title'][:90]}"
        )


        print(
            f"   Channel: "
            f"{item['channel']}"
        )


        print(
            f"   Duration: "
            f"{duration / 60:.1f} min"
        )


        print(
            f"   Views: "
            f"{views:,}"
        )


    return result


# ============================================================
# VIDEO DETAILS
# ============================================================

def get_video_details(
    video_ids,
    api_key,
):

    params = {

        "part": (
            "snippet,"
            "contentDetails,"
            "statistics"
        ),

        "id": ",".join(
            video_ids
        ),

        "key": api_key,
    }


    response = requests.get(

        f"{YOUTUBE_API_URL}/videos",

        params=params,

        timeout=30,
    )


    if response.status_code != 200:

        raise RuntimeError(

            "YouTube videos.list error: "

            f"{response.status_code}\n"

            f"{response.text[:2000]}"
        )


    data = response.json()


    details = {}


    for item in data.get(
        "items",
        []
    ):

        video_id = item.get(
            "id"
        )


        duration = parse_iso_duration(

            item
            .get("contentDetails", {})
            .get("duration", "")
        )


        views = int(

            item
            .get("statistics", {})
            .get("viewCount", 0)
        )


        details[video_id] = {

            "duration": duration,

            "views": views,

        }


    return details


# ============================================================
# ISO 8601 DURATION
# ============================================================

def parse_iso_duration(
    value
):

    if not value:

        return 0


    match = re.fullmatch(

        r"PT"
        r"(?:(\d+)H)?"
        r"(?:(\d+)M)?"
        r"(?:(\d+)S)?",

        value,
    )


    if not match:

        return 0


    hours = int(
        match.group(1) or 0
    )


    minutes = int(
        match.group(2) or 0
    )


    seconds = int(
        match.group(3) or 0
    )


    return (

        hours * 3600

        + minutes * 60

        + seconds
    )


# ============================================================
# YT-DLP
# ============================================================

def download_video(
    video
):

    url = video["url"]


    print("")
    print("=" * 70)
    print("⬇️ TRYING YT-DLP")
    print("=" * 70)


    print(
        f"Title: {video['title']}"
    )


    print(
        f"URL: {url}"
    )


    # --------------------------------------------------------
    # Remove old files
    # --------------------------------------------------------

    for file in Path("input").glob(
        "source.*"
    ):

        try:

            file.unlink()

        except Exception:
            pass


    # --------------------------------------------------------
    # Format
    # --------------------------------------------------------
    #
    # Prefer <=720p.
    #
    # Prefer MP4/M4A.
    #
    # If separate video/audio streams exist,
    # ffmpeg merges them into source.mp4.
    #
    # If a combined format exists, use it as fallback.
    #
    # This avoids the previous 4.2 GB 4K download.
    # --------------------------------------------------------

    format_selector = (

        "((bv*[height<=720][ext=mp4]"
        "+ba[ext=m4a])"
        "/(bv*[height<=720]"
        "+ba)"
        "/b[height<=720])"

    )


    output_template = (
        "input/source.%(ext)s"
    )


    command = [

        sys.executable,

        "-m",

        "yt_dlp",

        "--no-playlist",

        "--no-part",

        "--no-overwrites",

        "--js-runtimes",
        "deno",

        "--remote-components",
        "ejs:github",

        "--format",
        format_selector,

        "--merge-output-format",
        "mp4",

        "--output",
        output_template,

        "--retries",
        "3",

        "--fragment-retries",
        "3",

        "--socket-timeout",
        "30",

        "--no-warnings",

        url,
    ]


    print("")
    print(
        "Running:"
    )


    print(
        " ".join(command)
    )


    try:

        result = subprocess.run(

            command,

            stdout=None,

            stderr=None,

            text=True,

            timeout=DOWNLOAD_TIMEOUT,
        )


    except subprocess.TimeoutExpired:

        print(
            "❌ yt-dlp timeout"
        )

        return False


    if result.returncode != 0:

        print("")
        print(
            "❌ yt-dlp failed"
        )

        return False


    # --------------------------------------------------------
    # Find resulting MP4
    # --------------------------------------------------------

    source_files = list(
        Path("input").glob(
            "source.*"
        )
    )


    mp4_files = [

        file

        for file in source_files

        if file.suffix.lower() == ".mp4"

    ]


    if not mp4_files:

        print(
            "❌ yt-dlp finished "
            "but source.mp4 was not created"
        )

        return False


    source = mp4_files[0]


    # Ensure exact expected filename.

    if source != SOURCE_VIDEO:

        if SOURCE_VIDEO.exists():

            SOURCE_VIDEO.unlink()


        source.rename(
            SOURCE_VIDEO
        )


    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not SOURCE_VIDEO.exists():

        return False


    size_mb = (

        SOURCE_VIDEO.stat().st_size

        / 1024

        / 1024

    )


    print("")
    print("=" * 70)
    print("✅ SOURCE VIDEO DOWNLOADED")
    print("=" * 70)


    print(
        f"File: {SOURCE_VIDEO}"
    )


    print(
        f"Size: {size_mb:.1f} MB"
    )


    return True


# ============================================================
# FIND DOWNLOADABLE VIDEO
# ============================================================

def find_and_download():

    candidates = search_youtube()


    print("")
    print("=" * 70)
    print("🎯 DOWNLOAD CANDIDATES")
    print("=" * 70)


    # Try up to all returned candidates.
    #
    # This is important:
    # if YouTube blocks one video,
    # we don't kill the entire workflow.

    for index, video in enumerate(
        candidates,
        start=1
    ):

        print("")
        print(
            f"Candidate {index}/"
            f"{len(candidates)}"
        )


        success = download_video(
            video
        )


        if success:

            return


        print("")
        print(
            "⚠️ This video could not "
            "be downloaded."
        )


    raise RuntimeError(

        "❌ None of the YouTube "
        "candidates could be downloaded."
    )


# ============================================================
# CLIP INFO
# ============================================================

def print_clip_info(
    clips
):

    print("")
    print("=" * 70)
    print("🔥 SELECTED SHORTS")
    print("=" * 70)


    for index, clip in enumerate(
        clips,
        start=1
    ):

        start = float(
            clip["start"]
        )


        end = float(
            clip["end"]
        )


        duration = end - start


        score = clip.get(
            "score",
            0
        )


        print("")

        print(
            f"#{index}"
        )


        print(
            f"⏱ {start:.2f}s → "
            f"{end:.2f}s"
        )


        print(
            f"⌛ Duration: "
            f"{duration:.2f}s"
        )


        print(
            f"⭐ Score: "
            f"{score}"
        )


        print(
            f"🎯 Hook: "
            f"{clip.get('hook', '')}"
        )


        print(
            f"💡 Reason: "
            f"{clip.get('reason', '')}"
        )


        print(
            f"📝 Title hint: "
            f"{clip.get('title_hint', '')}"
        )


    print("")


# ============================================================
# AI PROCESSING
# ============================================================

def process_video():

    print_header()


    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    if not os.getenv(
        "GEMINI_KEY"
    ):

        raise RuntimeError(
            "GEMINI_KEY environment "
            "variable is not set"
        )


    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    print(
        "📹 STEP 1/6 — "
        "Checking source video"
    )


    validate_source_video(
        SOURCE_VIDEO
    )


    duration = get_duration(
        SOURCE_VIDEO
    )


    print(
        f"⏱ Source duration: "
        f"{duration / 60:.2f} minutes"
    )


    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    print("")

    print(
        "🎧 STEP 2/6 — "
        "Extracting audio"
    )


    extract_audio(

        SOURCE_VIDEO,

        AUDIO_FILE,
    )


    # --------------------------------------------------------
    # WHISPER
    # --------------------------------------------------------

    print("")

    print(
        "🎙 STEP 3/6 — "
        "Transcribing with Whisper"
    )


    transcription = (
        transcribe_with_language(
            AUDIO_FILE
        )
    )


    language = transcription[
        "language"
    ]


    words = transcription[
        "words"
    ]


    if not words:

        raise RuntimeError(
            "Whisper returned no words"
        )


    print("")

    print(
        f"🌍 Detected language: "
        f"{language}"
    )


    print(
        f"📝 Word count: "
        f"{len(words)}"
    )


    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    print("")

    print(
        "🤖 STEP 4/6 — "
        "Finding the best Shorts"
    )


    clips = select_clips(

        words,

        max_clips=MAX_CLIPS,
    )


    if not clips:

        raise RuntimeError(

            "Gemini did not find "
            "suitable clips"
        )


    print_clip_info(
        clips
    )


    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    print("")

    print(
        "🎬 STEP 5/6 — "
        "Rendering Shorts"
    )


    rendered = render_clips(

        source_video=SOURCE_VIDEO,

        clips=clips,

        output_dir=CLIPS_DIR,
    )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("")

    print(
        "📦 STEP 6/6 — "
        "Pipeline complete"
    )


    print("")

    print("=" * 70)

    print(
        "✅ GENERATED SHORTS"
    )

    print("=" * 70)


    for path in rendered:

        print(
            f"🎬 {path}"
        )


    print("")

    print("=" * 70)

    print(
        "🎉 DONE"
    )

    print("=" * 70)

    print("")


# ============================================================
# MAIN
# ============================================================

def main():

    if "--download" in sys.argv:

        find_and_download()

        return


    if "--process" in sys.argv:

        process_video()

        return


    # Default behavior:
    # process already downloaded source.

    process_video()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
