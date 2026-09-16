import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


WIDTH = 1080
HEIGHT = 1920

# Анализ лица каждые 0.20 секунды.
TRACK_INTERVAL = 0.20

# Минимальная уверенность детектора.
FACE_CONFIDENCE = 0.45

# Насколько плавно камера следует за лицом.
SMOOTHING = 0.20

# Ограничение резкого движения камеры.
MAX_MOVE_RATIO = 0.035

# Немного поднимаем лицо относительно центра.
# Отрицательное значение = выше.
FACE_VERTICAL_BIAS = -0.08


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

PROTOTXT = (
    MODEL_DIR
    / "deploy.prototxt"
)

MODEL = (
    MODEL_DIR
    / "res10_300x300_ssd_iter_140000.caffemodel"
)


# ============================================================
# FFMPEG
# ============================================================

def _run_ffmpeg(command):

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
            "FFmpeg failed with exit code "
            f"{result.returncode}"
        )

    return result


# ============================================================
# VIDEO INFO
# ============================================================

def _get_video_dimensions(video_path):

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
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
            "Unable to inspect video:\n"
            f"{result.stderr}"
        )

    data = json.loads(
        result.stdout
    )

    streams = data.get(
        "streams",
        [],
    )

    if not streams:

        raise RuntimeError(
            "No video stream found"
        )

    return (
        int(streams[0]["width"]),
        int(streams[0]["height"]),
    )


# ============================================================
# LOAD DNN FACE DETECTOR
# ============================================================

def _load_face_detector():

    print("")
    print("🧠 Loading DNN face detector...")

    if not PROTOTXT.exists():

        raise FileNotFoundError(
            "Missing face detector config:\n"
            f"{PROTOTXT}\n\n"
            "Create models/deploy.prototxt"
        )

    if not MODEL.exists():

        raise FileNotFoundError(
            "Missing face detector model:\n"
            f"{MODEL}\n\n"
            "Download "
            "res10_300x300_ssd_iter_140000.caffemodel "
            "and put it into models/"
        )

    print(
        f"📄 Config: {PROTOTXT}"
    )

    print(
        f"🧠 Model: {MODEL}"
    )

    detector = cv2.dnn.readNetFromCaffe(
        str(PROTOTXT),
        str(MODEL),
    )

    detector.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    detector.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    print(
        "✅ DNN face detector loaded"
    )

    return detector


# ============================================================
# DETECT FACES
# ============================================================

def _detect_faces(
    frame,
    detector,
):
    """
    Detect faces using OpenCV DNN SSD.
    """

    height, width = (
        frame.shape[:2]
    )

    blob = cv2.dnn.blobFromImage(
        cv2.resize(
            frame,
            (300, 300),
        ),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )

    detector.setInput(
        blob
    )

    detections = detector.forward()

    faces = []

    for i in range(
        detections.shape[2]
    ):

        confidence = float(
            detections[
                0,
                0,
                i,
                2,
            ]
        )

        if confidence < FACE_CONFIDENCE:
            continue

        box = (
            detections[
                0,
                0,
                i,
                3:7,
            ]
            * np.array(
                [
                    width,
                    height,
                    width,
                    height,
                ]
            )
        )

        x1, y1, x2, y2 = (
            box.astype(int)
        )

        x1 = max(
            0,
            min(
                x1,
                width - 1,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                height - 1,
            ),
        )

        x2 = max(
            0,
            min(
                x2,
                width - 1,
            ),
        )

        y2 = max(
            0,
            min(
                y2,
                height - 1,
            ),
        )

        face_width = x2 - x1
        face_height = y2 - y1

        if (
            face_width <= 0
            or face_height <= 0
        ):
            continue

        faces.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": face_width,
                "height": face_height,
                "confidence": confidence,
                "center_x": (
                    x1 + x2
                ) / 2,
                "center_y": (
                    y1 + y2
                ) / 2,
                "area": (
                    face_width
                    * face_height
                ),
            }
        )

    return faces


# ============================================================
# CHOOSE FACE
# ============================================================

def _choose_face(
    faces,
    previous_center,
    source_width,
    source_height,
):
    """
    Select the speaker face.

    First frame:
        largest/highest confidence face.

    Following frames:
        prefer face closest to previous position.
    """

    if not faces:
        return None

    # First detection.
    if previous_center is None:

        return max(
            faces,
            key=lambda face: (
                face["area"]
                * face["confidence"]
            ),
        )

    previous_x, previous_y = (
        previous_center
    )

    def score(face):

        dx = (
            face["center_x"]
            - previous_x
        ) / source_width

        dy = (
            face["center_y"]
            - previous_y
        ) / source_height

        distance = (
            dx * dx
            + dy * dy
        )

        area_ratio = (
            face["area"]
            / (
                source_width
                * source_height
            )
        )

        confidence = (
            face["confidence"]
        )

        return (
            distance * 5.0
            - area_ratio * 1.5
            - confidence * 0.25
        )

    return min(
        faces,
        key=score,
    )


# ============================================================
# TRACK FACE
# ============================================================

def _track_face(
    source_video,
    start,
    end,
    source_width,
    source_height,
):
    """
    Detect face every TRACK_INTERVAL seconds.
    """

    detector = (
        _load_face_detector()
    )

    capture = cv2.VideoCapture(
        str(source_video)
    )

    if not capture.isOpened():

        raise RuntimeError(
            "Unable to open source video "
            "for face tracking"
        )

    duration = (
        float(end)
        - float(start)
    )

    timestamps = np.arange(
        0.0,
        duration,
        TRACK_INTERVAL,
    )

    if (
        len(timestamps) == 0
        or timestamps[-1] < duration
    ):

        timestamps = np.append(
            timestamps,
            duration,
        )

    tracked = []

    previous_center = None

    detected_count = 0

    for relative_time in timestamps:

        absolute_time = (
            float(start)
            + float(relative_time)
        )

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            absolute_time * 1000.0,
        )

        ok, frame = (
            capture.read()
        )

        if not ok or frame is None:

            continue

        faces = _detect_faces(
            frame,
            detector,
        )

        selected = _choose_face(
            faces,
            previous_center,
            source_width,
            source_height,
        )

        if selected is None:

            if previous_center is not None:

                tracked.append(
                    (
                        float(relative_time),
                        previous_center[0],
                        previous_center[1],
                    )
                )

            continue

        detected_count += 1

        center_x = (
            selected["center_x"]
        )

        center_y = (
            selected["center_y"]
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
        f"👤 Face detections: "
        f"{detected_count}/"
        f"{len(timestamps)}"
    )

    return tracked


# ============================================================
# SMOOTH TRACKING
# ============================================================

def _smooth_tracking(
    tracked,
    source_width,
    source_height,
):
    """
    Smooth camera movement.
    """

    if not tracked:
        return []

    smoothed = []

    current_x = tracked[0][1]
    current_y = tracked[0][2]

    max_x_move = (
        source_width
        * MAX_MOVE_RATIO
    )

    max_y_move = (
        source_height
        * MAX_MOVE_RATIO
    )

    for (
        timestamp,
        target_x,
        target_y,
    ) in tracked:

        dx = (
            target_x
            - current_x
        )

        dy = (
            target_y
            - current_y
        )

        dx = max(
            -max_x_move,
            min(
                dx,
                max_x_move,
            ),
        )

        dy = max(
            -max_y_move,
            min(
                dy,
                max_y_move,
            ),
        )

        current_x += (
            dx
            * SMOOTHING
        )

        current_y += (
            dy
            * SMOOTHING
        )

        smoothed.append(
            (
                timestamp,
                current_x,
                current_y,
            )
        )

    return smoothed


# ============================================================
# BUILD PIECEWISE FFmpeg EXPRESSION
# ============================================================

def _build_expression(
    points,
    fallback,
):
    """
    Convert tracking points into FFmpeg
    piecewise-linear expression.
    """

    if not points:

        return str(
            int(round(fallback))
        )

    expression = str(
        int(round(points[-1][1]))
    )

    for i in range(
        len(points) - 1,
        0,
        -1,
    ):

        t1, p1 = points[i - 1]
        t2, p2 = points[i]

        if t2 <= t1:
            continue

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


# ============================================================
# FACE-AWARE CROP
# ============================================================

def _create_face_crop(
    source_video,
    start,
    end,
    source_width,
    source_height,
):
    """
    Create dynamic crop based on face position.
    """

    # 9:16 crop.
    target_ratio = (
        9 / 16
    )

    source_ratio = (
        source_width
        / source_height
    )

    if source_ratio > target_ratio:

        crop_height = (
            source_height
        )

        crop_width = int(
            round(
                crop_height
                * target_ratio
            )
        )

    else:

        crop_width = (
            source_width
        )

        crop_height = int(
            round(
                crop_width
                / target_ratio
            )
        )

    crop_width = min(
        crop_width,
        source_width,
    )

    crop_height = min(
        crop_height,
        source_height,
    )

    max_x = (
        source_width
        - crop_width
    )

    max_y = (
        source_height
        - crop_height
    )

    print("")
    print(
        f"📐 Crop area: "
        f"{crop_width}x"
        f"{crop_height}"
    )

    tracked = _track_face(
        source_video,
        start,
        end,
        source_width,
        source_height,
    )

    if not tracked:

        print(
            "⚠️ Face not detected"
        )

        print(
            "↩️ Falling back to center crop"
        )

        return (
            f"crop="
            f"{crop_width}:"
            f"{crop_height}:"
            f"{int(max_x / 2)}:"
            f"{int(max_y / 2)}"
        )

    smoothed = _smooth_tracking(
        tracked,
        source_width,
        source_height,
    )

    print(
        "🎯 Face tracking ACTIVE"
    )

    print(
        f"🎯 Tracking points: "
        f"{len(smoothed)}"
    )

    x_points = []
    y_points = []

    for (
        timestamp,
        center_x,
        center_y,
    ) in smoothed:

        # Move face slightly upward
        # in the final vertical composition.
        center_y += (
            crop_height
            * FACE_VERTICAL_BIAS
        )

        crop_x = (
            center_x
            - crop_width / 2
        )

        crop_y = (
            center_y
            - crop_height / 2
        )

        crop_x = max(
            0,
            min(
                crop_x,
                max_x,
            ),
        )

        crop_y = max(
            0,
            min(
                crop_y,
                max_y,
            ),
        )

        x_points.append(
            (
                timestamp,
                crop_x,
            )
        )

        y_points.append(
            (
                timestamp,
                crop_y,
            )
        )

    initial_x = x_points[0][1]
    initial_y = y_points[0][1]

    x_expression = (
        _build_expression(
            x_points,
            initial_x,
        )
    )

    y_expression = (
        _build_expression(
            y_points,
            initial_y,
        )
    )

    print(
        f"🎯 Initial X: "
        f"{initial_x:.1f}"
    )

    print(
        f"🎯 Initial Y: "
        f"{initial_y:.1f}"
    )

    return (
        f"crop="
        f"{crop_width}:"
        f"{crop_height}:"
        f"{x_expression}:"
        f"{y_expression}"
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
    Render one 9:16 Short with face tracking.
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

    duration = (
        end - start
    )

    if duration <= 0:

        raise ValueError(
            "Invalid clip duration"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("")
    print(
        "🎞 Creating Short"
    )

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
        "Face tracking: DNN"
    )

    source_width, source_height = (
        _get_video_dimensions(
            source_video
        )
    )

    print(
        f"📐 Source resolution: "
        f"{source_width}x"
        f"{source_height}"
    )

    crop_filter = (
        _create_face_crop(
            source_video,
            start,
            end,
            source_width,
            source_height,
        )
    )

    # Scale after crop.
    video_filter = (
        crop_filter
        + ","
        + "scale=1080:1920"
        + ":flags=lanczos"
    )

    print("")
    print(
        "🎥 FFmpeg filter:"
    )

    print(
        video_filter
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
        video_filter,

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
            "Output video was not created"
        )

    size_mb = (
        output_path.stat().st_size
        / (
            1024 * 1024
        )
    )

    print("")
    print(
        "✅ Short created"
    )

    print(
        f"📁 {output_path}"
    )

    print(
        f"💾 {size_mb:.2f} MB"
    )

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
    """
    Render all selected clips.
    """

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

        render_clip(
            source_video,
            output_path,
            start,
            end,
        )

        rendered.append(
            str(output_path)
        )

    return rendered
