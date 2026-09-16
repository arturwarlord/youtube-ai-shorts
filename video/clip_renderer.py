import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


WIDTH = 1080
HEIGHT = 1920

# Face detection is only used to choose the vertical crop. FFmpeg still
# performs the actual encoding, subtitles, audio handling, and output.
FACE_SAMPLE_COUNT = 12
FACE_SCALE = 0.5
FACE_MIN_NEIGHBORS = 4
FACE_MIN_SIZE = 32
FACE_VERTICAL_BIAS = -0.08


def _run_ffmpeg(command):
    """Run FFmpeg command and raise a readable error if it fails."""

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
    """Get video duration using ffprobe."""

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


def _center_crop_box(source_width, source_height, target_ratio=9 / 16):
    """Return a centered crop rectangle with the requested aspect ratio."""

    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        crop_height = source_height
        crop_width = int(round(crop_height * target_ratio))
    else:
        crop_width = source_width
        crop_height = int(round(crop_width / target_ratio))

    crop_width = min(crop_width, source_width)
    crop_height = min(crop_height, source_height)

    x = max(0, (source_width - crop_width) // 2)
    y = max(0, (source_height - crop_height) // 2)

    return x, y, crop_width, crop_height


def _detect_face_crop(source_video, start, end, source_width, source_height):
    """
    Detect the dominant face in sampled frames and place the 9:16 crop
    around it. If no face is detected, fall back to the original center crop.

    The crop is intentionally static for the whole clip: this avoids
    aggressive camera movement/jitter while still keeping a speaker's face
    in frame instead of blindly cropping the landscape center.
    """

    fallback = _center_crop_box(source_width, source_height)

    cascade_path = getattr(
        cv2.data,
        "haarcascades",
        "",
    )
    cascade_path = str(
        Path(cascade_path) / "haarcascade_frontalface_default.xml"
    )

    if not Path(cascade_path).exists():
        print("⚠️ OpenCV Haar cascade not found; using center crop.")
        return fallback

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        print("⚠️ OpenCV face detector could not be loaded; using center crop.")
        return fallback

    capture = cv2.VideoCapture(str(source_video))

    if not capture.isOpened():
        print("⚠️ Could not open source for face detection; using center crop.")
        return fallback

    clip_duration = max(0.1, float(end) - float(start))
    sample_times = np.linspace(0.0, clip_duration, FACE_SAMPLE_COUNT)

    detected_faces = []

    for relative_time in sample_times:
        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            (float(start) + float(relative_time)) * 1000.0,
        )

        ok, frame = capture.read()

        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if FACE_SCALE != 1.0:
            small = cv2.resize(
                gray,
                None,
                fx=FACE_SCALE,
                fy=FACE_SCALE,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = gray

        faces = detector.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE),
        )

        if len(faces) == 0:
            continue

        # Convert detections back to source-video coordinates.
        scale_back = 1.0 / FACE_SCALE
        scaled_faces = []

        for x, y, width, height in faces:
            x = int(round(x * scale_back))
            y = int(round(y * scale_back))
            width = int(round(width * scale_back))
            height = int(round(height * scale_back))
            area = width * height
            scaled_faces.append((x, y, width, height, area))

        # Largest visible face is the most useful fallback for interviews.
        scaled_faces.sort(key=lambda item: item[4], reverse=True)
        detected_faces.append(scaled_faces[0])

    capture.release()

    if not detected_faces:
        print("👤 No face detected; using center crop.")
        return fallback

    # Use robust medians so one missed/incorrect frame does not move the crop.
    centers_x = np.array([x + width / 2 for x, y, width, height, _ in detected_faces])
    centers_y = np.array([y + height / 2 for x, y, width, height, _ in detected_faces])

    center_x = float(np.median(centers_x))
    center_y = float(np.median(centers_y))

    crop_x, crop_y, crop_width, crop_height = fallback

    # Slight upward bias keeps eyes/head higher in the vertical frame and
    # leaves more room below for subtitles.
    center_y += crop_height * FACE_VERTICAL_BIAS

    crop_x = int(round(center_x - crop_width / 2))
    crop_y = int(round(center_y - crop_height / 2))

    crop_x = max(0, min(crop_x, source_width - crop_width))
    crop_y = max(0, min(crop_y, source_height - crop_height))

    print(
        f"👤 Face-aware crop: x={crop_x}, y={crop_y}, "
        f"w={crop_width}, h={crop_height}, "
        f"samples={len(detected_faces)}"
    )

    return crop_x, crop_y, crop_width, crop_height


def render_clip(
    source_video,
    output_path,
    start,
    end,
):
    """Cut a segment from a source video and convert it to 9:16."""

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

    crop_x, crop_y, crop_width, crop_height = _detect_face_crop(
        source_video,
        start,
        end,
        source_width,
        source_height,
    )

    # Fixed crop dimensions + fixed position. This is compatible with
    # every ffmpeg build and avoids center-cropping faces away.
    crop_filter = (
        f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(source_video),
        "-t",
        str(duration),
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
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    _run_ffmpeg(command)

    if not output_path.exists():
        raise RuntimeError(
            f"FFmpeg finished but output was not created: {output_path}"
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
    """Render multiple selected clips."""

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
