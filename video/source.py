```python
import subprocess
from pathlib import Path


def run_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"Command failed with code {result.returncode}"
        )

    return result


def validate_source_video(video_path):
    """
    Check that the source video exists and is readable.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Source video not found: {video_path}"
        )

    if video_path.stat().st_size == 0:
        raise RuntimeError(
            f"Source video is empty: {video_path}"
        )

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,duration",
        "-of",
        "json",
        str(video_path),
    ]

    result = run_command(command)

    print("✅ Source video is valid")

    return result.stdout


def get_duration(video_path):
    """
    Return video duration in seconds.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = run_command(command)

    return float(result.stdout.strip())


def extract_audio(video_path, audio_path):
    """
    Extract mono 16 kHz WAV audio for Whisper.
    """

    video_path = Path(video_path)
    audio_path = Path(audio_path)

    audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("")
    print("🎧 Extracting audio for Whisper...")
    print(f"Source: {video_path}")
    print(f"Audio:  {audio_path}")

    command = [
        "ffmpeg",
        "-y",

        "-i",
        str(video_path),

        "-vn",

        "-ac",
        "1",

        "-ar",
        "16000",

        "-c:a",
        "pcm_s16le",

        str(audio_path),
    ]

    run_command(command)

    if not audio_path.exists():
        raise RuntimeError(
            f"Audio extraction failed: {audio_path}"
        )

    print("✅ Audio extracted")

    return str(audio_path)
```

