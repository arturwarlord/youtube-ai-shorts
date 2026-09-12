```python
import os
from pathlib import Path

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


# ============================================================
# HELPERS
# ============================================================

def print_header():
    print("")
    print("=" * 70)
    print("🚀 AI LONG VIDEO → YOUTUBE SHORTS")
    print("=" * 70)
    print("")


def print_clip_info(clips):
    print("")
    print("=" * 70)
    print("🔥 SELECTED SHORTS")
    print("=" * 70)

    for index, clip in enumerate(clips, start=1):
        start = float(clip["start"])
        end = float(clip["end"])
        duration = end - start
        score = clip.get("score", 0)

        print("")
        print(f"#{index}")
        print(f"⏱ {start:.2f}s → {end:.2f}s")
        print(f"⌛ Duration: {duration:.2f}s")
        print(f"⭐ Score: {score}")

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
# MAIN PIPELINE
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # 1. Check environment
    # --------------------------------------------------------

    if not os.getenv("GEMINI_KEY"):
        raise RuntimeError(
            "GEMINI_KEY environment variable is not set"
        )

    # --------------------------------------------------------
    # 2. Check source video
    # --------------------------------------------------------

    print("📹 STEP 1/6 — Checking source video")

    validate_source_video(SOURCE_VIDEO)

    duration = get_duration(SOURCE_VIDEO)

    print(
        f"⏱ Source duration: "
        f"{duration / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # 3. Extract audio
    # --------------------------------------------------------

    print("")
    print("🎧 STEP 2/6 — Extracting audio")

    extract_audio(
        SOURCE_VIDEO,
        AUDIO_FILE,
    )

    # --------------------------------------------------------
    # 4. Whisper transcription
    # --------------------------------------------------------

    print("")
    print("🎙 STEP 3/6 — Transcribing with Whisper")

    transcription = transcribe_with_language(
        AUDIO_FILE
    )

    language = transcription["language"]
    words = transcription["words"]

    if not words:
        raise RuntimeError(
            "Whisper returned no words"
        )

    print("")
    print(f"🌍 Detected language: {language}")
    print(f"📝 Word count: {len(words)}")

    # --------------------------------------------------------
    # 5. Gemini clip selection
    # --------------------------------------------------------

    print("")
    print("🤖 STEP 4/6 — Finding the best Shorts")

    clips = select_clips(
        words,
        max_clips=MAX_CLIPS,
    )

    if not clips:
        raise RuntimeError(
            "Gemini did not find suitable clips"
        )

    print_clip_info(clips)

    # --------------------------------------------------------
    # 6. Render clips
    # --------------------------------------------------------

    print("")
    print("🎬 STEP 5/6 — Rendering Shorts")

    rendered = render_clips(
        source_video=SOURCE_VIDEO,
        clips=clips,
        output_dir=CLIPS_DIR,
    )

    # --------------------------------------------------------
    # 7. Final result
    # --------------------------------------------------------

    print("")
    print("📦 STEP 6/6 — Pipeline complete")

    print("")
    print("=" * 70)
    print("✅ GENERATED SHORTS")
    print("=" * 70)

    for path in rendered:
        print(f"🎬 {path}")

    print("")
    print("=" * 70)
    print("🎉 DONE")
    print("=" * 70)
    print("")


if __name__ == "__main__":
    main()
```
