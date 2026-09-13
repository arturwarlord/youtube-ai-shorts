import os
import subprocess
import json
from pathlib import Path


WIDTH = 1080
HEIGHT = 1920


def _run_ffmpeg(command):
    """
    Run FFmpeg command and raise a readable error if it fails.
    """

    print("🎬 Running FFmpeg...")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"FFmpeg failed with exit code {result.returncode}"
        )

    return result


def get_video_duration(video_path):
    """
    Get video duration using ffprobe.
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

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Unable to read video duration: {result.stderr}"
        )

    return float(result.stdout.strip())


def render_clip(
    source_video,
    output_path,
    start,
    end,
):
    """
    Cut a segment from a source video and convert it to 9:16.

    The original source audio is preserved.
    """

    source_video = Path(source_video)
    output_path = Path(output_path)

    if not source_video.exists():
        raise FileNotFoundError(
            f"Source video not found: {source_video}"
        )

    start = float(start)
    end = float(end)

    if end <= start:
        raise ValueError(
            f"Invalid clip range: {start} -> {end}"
        )

    duration = end - start

    if duration < 1:
        raise ValueError(
            f"Clip is too short: {duration:.2f}s"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("")
    print("🎞 Creating Short")
    print(f"Source: {source_video}")
    print(f"Start:  {start:.2f}s")
    print(f"End:    {end:.2f}s")
    print(f"Length: {duration:.2f}s")
    print("Format: 1080x1920")
    print("Audio: original")
    print("")

    # ---------------------------------------------------------
    # Get source dimensions using ffprobe JSON.
    # This is more reliable than parsing CSV output.
    # ---------------------------------------------------------

    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(source_video),
    ]

    probe_result = subprocess.run(
        probe_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if probe_result.returncode != 0:
        raise RuntimeError(
            f"Unable to inspect source video:\n"
            f"{probe_result.stderr}"
        )

    try:
        probe_data = json.loads(probe_result.stdout)

        streams = probe_data.get("streams", [])

        if not streams:
            raise ValueError("No video stream found")

        source_width = int(streams[0]["width"])
        source_height = int(streams[0]["height"])

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Unable to determine source dimensions.\n"
            f"ffprobe output:\n{probe_result.stdout}\n"
            f"Error: {error}"
        )

    print(
        f"📐 Source resolution: "
        f"{source_width}x{source_height}"
    )

    # ---------------------------------------------------------
    # Center crop to 9:16.
    # ---------------------------------------------------------

    crop_filter = (
        "crop="
        "if(gt(iw/ih\\,9/16)\\,ih*9/16\\,iw):"
        "if(gt(iw/ih\\,9/16)\\,ih\\,iw*16/9):"
        "(iw-ow)/2:"
        "(ih-oh)/2,"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    )

    command = [
        "ffmpeg",
        "-y",

        # Accurate seeking.
        "-ss",
        str(start),

        "-i",
        str(source_video),

        "-t",
        str(duration),

        # Video.
        "-vf",
        crop_filter,

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "20",

        "-pix_fmt",
        "yuv420p",

        # Original audio.
        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-ar",
        "48000",

        # Fast start for web playback.
        "-movflags",
        "+faststart",

        str(output_path),
    ]

    _run_ffmpeg(command)

    if not output_path.exists():
        raise RuntimeError(
            f"FFmpeg finished but output was not created: "
            f"{output_path}"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print("")
    print("✅ Short created")
    print(f"📁 {output_path}")
    print(f"💾 {size_mb:.2f} MB")
    print("")

    return str(output_path)


def render_clips(
    source_video,
    clips,
    output_dir="output/clips",
):
    """
    Render multiple selected clips.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered = []

    for index, clip in enumerate(clips, start=1):

        start = float(clip["start"])
        end = float(clip["end"])

        output_path = (
            output_dir /
            f"clip_{index:02d}.mp4"
        )

        print("")
        print("=" * 60)
        print(
            f"🎬 Rendering clip "
            f"{index}/{len(clips)}"
        )
        print("=" * 60)

        render_clip(
            source_video=source_video,
            output_path=output_path,
            start=start,
            end=end,
        )

        rendered.append(str(output_path))

    return rendered
