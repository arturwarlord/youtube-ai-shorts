import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


WIDTH = 1080
HEIGHT = 1920

# ============================================================
# FACE TRACKING SETTINGS
# ============================================================

# Анализируем примерно 1 кадр каждые N секунд.
TRACK_INTERVAL = 0.20

# Уменьшаем кадр перед распознаванием лица для скорости.
FACE_SCALE = 0.5

# Параметры Haar detector.
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = 40

# Насколько сильно камера может двигаться за один шаг.
# Чем меньше значение — тем плавнее движение.
MAX_MOVE_PER_FRAME = 0.035

# Сглаживание движения камеры.
SMOOTHING = 0.18

# Лицо будет немного выше центра вертикального видео.
# Это оставляет место для субтитров снизу.
FACE_VERTICAL_BIAS = -0.08

# Минимальная доля лица относительно ширины кадра,
# чтобы не принимать очень маленькие ложные лица.
MIN_FACE_RATIO = 0.025


# ============================================================
# FFMPEG
# ============================================================

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


# ============================================================
# VIDEO INFO
# ============================================================

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


# ============================================================
# CENTER CROP
# ============================================================

def _center_crop_box(
    source_width,
    source_height,
    target_ratio=9 / 16,
):
    """Return centered 9:16 crop rectangle."""

    source_ratio = source_width / source_height

    if source_ratio > target_ratio:

        crop_height = source_height
        crop_width = int(
            round(crop_height * target_ratio)
        )

    else:

        crop_width = source_width
        crop_height = int(
            round(crop_width / target_ratio)
        )

    crop_width = min(
        crop_width,
        source_width,
    )

    crop_height = min(
        crop_height,
        source_height,
    )

    x = max(
        0,
        (source_width - crop_width) // 2,
    )

    y = max(
        0,
        (source_height - crop_height) // 2,
    )

    return (
        x,
        y,
        crop_width,
        crop_height,
    )


# ============================================================
# LOAD FACE DETECTORS
# ============================================================

def _load_face_detectors():
    """
    Load several OpenCV face detectors.

    We use both frontal and profile cascades because
    people in podcasts/interviews are not always looking
    directly into the camera.
    """

    haar_dir = Path(
        cv2.data.haarcascades
    )

    detectors = []

    detector_files = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]

    for filename in detector_files:

        path = haar_dir / filename

        if not path.exists():
            print(
                f"⚠️ Haar detector missing: {path}"
            )
            continue

        detector = cv2.CascadeClassifier(
            str(path)
        )

        if detector.empty():
            print(
                f"⚠️ Failed to load detector: {path}"
            )
            continue

        detectors.append(
            detector
        )

    return detectors


# ============================================================
# DETECT FACES
# ============================================================

def _detect_faces(
    frame,
    detectors,
    source_width,
):
    """
    Detect faces in a frame.

    Returns:
        list of:
        (x, y, width, height)
    """

    if frame is None:
        return []

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

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

    min_face_width = max(
        FACE_MIN_SIZE,
        int(source_width * MIN_FACE_RATIO * FACE_SCALE),
    )

    found = []

    for detector in detectors:

        try:

            faces = detector.detectMultiScale(
                small,
                scaleFactor=1.08,
                minNeighbors=FACE_MIN_NEIGHBORS,
                minSize=(
                    min_face_width,
                    min_face_width,
                ),
            )

        except Exception as error:

            print(
                f"⚠️ Face detector error: {error}"
            )

            continue

        if faces is None:
            continue

        scale_back = (
            1.0 / FACE_SCALE
        )

        for x, y, w, h in faces:

            x = int(
                round(x * scale_back)
            )

            y = int(
                round(y * scale_back)
            )

            w = int(
                round(w * scale_back)
            )

            h = int(
                round(h * scale_back)
            )

            if w <= 0 or h <= 0:
                continue

            found.append(
                (
                    x,
                    y,
                    w,
                    h,
                )
            )

    # Remove duplicate detections.
    unique = []

    for face in found:

        x, y, w, h = face

        duplicate = False

        for ux, uy, uw, uh in unique:

            cx = x + w / 2
            cy = y + h / 2

            ucx = ux + uw / 2
            ucy = uy + uh / 2

            distance = (
                (cx - ucx) ** 2
                + (cy - ucy) ** 2
            ) ** 0.5

            if distance < max(
                w,
                h,
                uw,
                uh,
            ) * 0.5:

                duplicate = True
                break

        if not duplicate:
            unique.append(face)

    return unique


# ============================================================
# CHOOSE FACE
# ============================================================

def _choose_face(
    faces,
    previous_center=None,
    source_width=None,
    source_height=None,
):
    """
    Select the most useful face.

    If we already have a tracked face,
    prefer the face closest to the previous position.

    Otherwise prefer the largest face.
    """

    if not faces:
        return None

    if (
        previous_center is None
        or source_width is None
        or source_height is None
    ):

        return max(
            faces,
            key=lambda f: f[2] * f[3],
        )

    previous_x, previous_y = previous_center

    def score(face):

        x, y, w, h = face

        center_x = x + w / 2
        center_y = y + h / 2

        distance = (
            (
                center_x - previous_x
            ) / source_width
        ) ** 2 + (
            (
                center_y - previous_y
            ) / source_height
        ) ** 2

        area = (
            w * h
        ) / (
            source_width * source_height
        )

        # Closer face gets better score.
        # Larger face also gets a bonus.
        return (
            distance * 4.0
            - area * 2.0
        )

    return min(
        faces,
        key=score,
    )


# ============================================================
# FACE TRACKING
# ============================================================

def _track_face(
    source_video,
    start,
    end,
    source_width,
    source_height,
):
    """
    Track the speaker's face during the entire clip.

    Returns a list of:
        (time, center_x, center_y)

    The list is later converted into a smooth
    FFmpeg crop expression.
    """

    detectors = _load_face_detectors()

    if not detectors:

        print(
            "❌ No OpenCV face detectors available."
        )

        return []

    capture = cv2.VideoCapture(
        str(source_video)
    )

    if not capture.isOpened():

        print(
            "❌ Could not open video for face tracking."
        )

        return []

    clip_duration = max(
        0.1,
        float(end) - float(start),
    )

    # Generate analysis timestamps.
    timestamps = np.arange(
        0.0,
        clip_duration,
        TRACK_INTERVAL,
    )

    # Always analyze the final moment.
    if (
        len(timestamps) == 0
        or timestamps[-1]
        < clip_duration
    ):

        timestamps = np.append(
            timestamps,
            clip_duration,
        )

    tracked = []

    previous_center = None

    for relative_time in timestamps:

        absolute_time = (
            float(start)
            + float(relative_time)
        )

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            absolute_time * 1000.0,
        )

        ok, frame = capture.read()

        if not ok or frame is None:
            continue

        faces = _detect_faces(
            frame,
            detectors,
            source_width,
        )

        selected = _choose_face(
            faces,
            previous_center,
            source_width,
            source_height,
        )

        if selected is None:

            # No detection on this frame.
            # Keep previous position.
            if previous_center is not None:

                tracked.append(
                    (
                        float(relative_time),
                        previous_center[0],
                        previous_center[1],
                    )
                )

            continue

        x, y, w, h = selected

        center_x = (
            x + w / 2
        )

        center_y = (
            y + h / 2
        )

        previous_center = (
            center_x,
            center_y,
        )

        tracked.append(
            (
                float(relative_time),
                center_x,
                center_y,
            )
        )

    capture.release()

    print(
        f"👤 Face tracking: "
        f"{len(tracked)} detections"
    )

    return tracked


# ============================================================
# SMOOTH TRACKING
# ============================================================

def _smooth_positions(
    tracked,
    source_width,
    source_height,
):
    """
    Smooth face movement.

    This prevents the vertical camera from jumping
    from left to right when the detector slightly
    changes the face position.
    """

    if not tracked:
        return []

    result = []

    current_x = tracked[0][1]
    current_y = tracked[0][2]

    for timestamp, target_x, target_y in tracked:

        delta_x = (
            target_x - current_x
        )

        delta_y = (
            target_y - current_y
        )

        # Limit sudden camera movement.
        max_x_move = (
            source_width
            * MAX_MOVE_PER_FRAME
        )

        max_y_move = (
            source_height
            * MAX_MOVE_PER_FRAME
        )

        delta_x = max(
            -max_x_move,
            min(
                delta_x,
                max_x_move,
            ),
        )

        delta_y = max(
            -max_y_move,
            min(
                delta_y,
                max_y_move,
            ),
        )

        current_x += (
            delta_x
            * SMOOTHING
        )

        current_y += (
            delta_y
            * SMOOTHING
        )

        result.append(
            (
                timestamp,
                current_x,
                current_y,
            )
        )

    return result


# ============================================================
# CREATE FACE-AWARE CROP
# ============================================================

def _detect_face_crop(
    source_video,
    start,
    end,
    source_width,
    source_height,
):
    """
    Detect and track the main face.

    Returns a crop expression.

    If face tracking fails completely,
    fall back to center crop.
    """

    fallback = _center_crop_box(
        source_width,
        source_height,
    )

    crop_x, crop_y, crop_width, crop_height = fallback

    tracked = _track_face(
        source_video,
        start,
        end,
        source_width,
        source_height,
    )

    if not tracked:

        print(
            "👤 No face detected."
        )

        print(
            "↩️ Using center crop."
        )

        return fallback, None

    smoothed = _smooth_positions(
        tracked,
        source_width,
        source_height,
    )

    if not smoothed:

        return fallback, None

    # --------------------------------------------------------
    # Calculate initial crop position.
    # --------------------------------------------------------

    first_x = smoothed[0][1]
    first_y = smoothed[0][2]

    first_y += (
        crop_height
        * FACE_VERTICAL_BIAS
    )

    initial_x = int(
        round(
            first_x
            - crop_width / 2
        )
    )

    initial_y = int(
        round(
            first_y
            - crop_height / 2
        )
    )

    initial_x = max(
        0,
        min(
            initial_x,
            source_width
            - crop_width,
        ),
    )

    initial_y = max(
        0,
        min(
            initial_y,
            source_height
            - crop_height,
        ),
    )

    print(
        f"🎯 Face tracking enabled"
    )

    print(
        f"🎯 Initial crop: "
        f"x={initial_x}, "
        f"y={initial_y}, "
        f"w={crop_width}, "
        f"h={crop_height}"
    )

    print(
        f"🎯 Tracking points: "
        f"{len(smoothed)}"
    )

    # --------------------------------------------------------
    # Convert tracking points into FFmpeg expressions.
    #
    # We use a piecewise linear interpolation based on
    # detected face positions.
    # --------------------------------------------------------

    x_points = []
    y_points = []

    max_x = (
        source_width
        - crop_width
    )

    max_y = (
        source_height
        - crop_height
    )

    for timestamp, center_x, center_y in smoothed:

        center_y += (
            crop_height
            * FACE_VERTICAL_BIAS
        )

        target_x = (
            center_x
            - crop_width / 2
        )

        target_y = (
            center_y
            - crop_height / 2
        )

        target_x = max(
            0,
            min(
                target_x,
                max_x,
            ),
        )

        target_y = max(
            0,
            min(
                target_y,
                max_y,
            ),
        )

        x_points.append(
            (
                float(timestamp),
                float(target_x),
            )
        )

        y_points.append(
            (
                float(timestamp),
                float(target_y),
            )
        )

    # --------------------------------------------------------
    # Build FFmpeg expressions.
    # --------------------------------------------------------

    def build_expression(points, fallback):

        if not points:
            return str(
                int(round(fallback))
            )

        expression = str(
            int(round(points[-1][1]))
        )

        for index in range(
            len(points) - 1,
            0,
            -1,
        ):

            t1, p1 = points[index - 1]
            t2, p2 = points[index]

            if t2 <= t1:
                continue

            # Linear interpolation.
            interpolation = (
                f"({p1:.3f}+"
                f"({p2:.3f}-{p1:.3f})*"
                f"(t-{t1:.3f})/"
                f"({t2:.3f}-{t1:.3f}))"
            )

            expression = (
                f"if(lt(t,{t2:.3f}),"
                f"{interpolation},"
                f"{expression})"
            )

        return expression

    x_expression = build_expression(
        x_points,
        initial_x,
    )

    y_expression = build_expression(
        y_points,
        initial_y,
    )

    crop_expression = (
        f"crop="
        f"{crop_width}:"
        f"{crop_height}:"
        f"{x_expression}:"
        f"{y_expression}"
    )

    return (
        (
            initial_x,
            initial_y,
            crop_width,
            crop_height,
        ),
        crop_expression,
    )


# ============================================================
# RENDER SINGLE CLIP
# ============================================================

def render_clip(
    source_video,
    output_path,
    start,
    end,
):
    """
    Cut a segment from a source video
    and convert it to 9:16 with face tracking.
    """

    source_video = Path(
        source_video
    )

    output_path = Path(
        output_path
    )

    if not source_video.exists():

        raise FileNotFoundError(
            f"Source video not found: "
            f"{source_video}"
        )

    start = float(start)
    end = float(end)

    if end <= start:

        raise ValueError(
            f"Invalid clip range: "
            f"{start} -> {end}"
        )

    duration = end - start

    if duration < 1:

        raise ValueError(
            f"Clip is too short: "
            f"{duration:.2f}s"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("")
    print("🎞 Creating Short")
    print(
        f"Source: {source_video}"
    )
    print(
        f"Start:  {start:.2f}s"
    )
    print(
        f"End:    {end:.2f}s"
    )
    print(
        f"Length: {duration:.2f}s"
    )
    print(
        "Format: 1080x1920"
    )
    print(
        "Audio: original"
    )
    print(
        "Face tracking: ENABLED"
    )
    print("")

    # ========================================================
    # GET SOURCE RESOLUTION
    # ========================================================

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
            "Unable to inspect source video:\n"
            f"{probe_result.stderr}"
        )

    try:

        probe_data = json.loads(
            probe_result.stdout
        )

        streams = probe_data.get(
            "streams",
            [],
        )

        if not streams:
            raise ValueError(
                "No video stream found"
            )

        source_width = int(
            streams[0]["width"]
        )

        source_height = int(
            streams[0]["height"]
        )

    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:

        raise RuntimeError(
            "Unable to determine "
            "source dimensions.\n"
            f"ffprobe output:\n"
            f"{probe_result.stdout}\n"
            f"Error: {error}"
        )

    print(
        f"📐 Source resolution: "
        f"{source_width}x{source_height}"
    )

    # ========================================================
    # FACE TRACKING
    # ========================================================

    (
        crop_box,
        dynamic_crop,
    ) = _detect_face_crop(
        source_video,
        start,
        end,
        source_width,
        source_height,
    )

    crop_x, crop_y, crop_width, crop_height = (
        crop_box
    )

    # If face tracking succeeded,
    # dynamic_crop contains time-based X/Y.
    if dynamic_crop:

        crop_filter = (
            dynamic_crop
            + ","
            "scale=1080:1920:"
            "force_original_aspect_ratio=decrease,"
            "pad=1080:1920:"
            "(ow-iw)/2:"
            "(oh-ih)/2"
        )

    else:

        crop_filter = (
            f"crop="
            f"{crop_width}:"
            f"{crop_height}:"
            f"{crop_x}:"
            f"{crop_y},"
            "scale=1080:1920:"
            "force_original_aspect_ratio=decrease,"
            "pad=1080:1920:"
            "(ow-iw)/2:"
            "(oh-ih)/2"
        )

    print(
        f"🎥 FFmpeg crop filter:"
    )

    print(
        crop_filter
    )

    # ========================================================
    # FFMPEG
    # ========================================================

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

    _run_ffmpeg(
        command
    )

    if not output_path.exists():

        raise RuntimeError(
            "FFmpeg finished but output "
            f"was not created: "
            f"{output_path}"
        )

    size_mb = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    print("")
    print("✅ Short created")
    print(
        f"📁 {output_path}"
    )
    print(
        f"💾 {size_mb:.2f} MB"
    )
    print("")

    return str(
        output_path
    )


# ============================================================
# RENDER MULTIPLE CLIPS
# ============================================================

def render_clips(
    source_video,
    clips,
    output_dir="output/clips",
):
    """Render multiple selected clips."""

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered = []

    for index, clip in enumerate(
        clips,
        start=1,
    ):

        start = float(
            clip["start"]
        )

        end = float(
            clip["end"]
        )

        output_path = (
            output_dir
            / f"clip_{index:02d}.mp4"
        )

        print("")
        print(
            "=" * 60
        )

        print(
            f"🎬 Rendering clip "
            f"{index}/{len(clips)}"
        )

        print(
            "=" * 60
        )

        render_clip(
            source_video=source_video,
            output_path=output_path,
            start=start,
            end=end,
        )

        rendered.append(
            str(output_path)
        )

    return rendered
