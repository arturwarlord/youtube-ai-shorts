import subprocess
from pathlib import Path

import cv2
import mediapipe as mp


# ============================================================
# OUTPUT SETTINGS
# ============================================================

WIDTH = 1080
HEIGHT = 1920

# How often to analyze the source video for a face
TRACK_INTERVAL = 0.25

# MediaPipe face detection confidence
MIN_DETECTION_CONFIDENCE = 0.45

# Position smoothing
SMOOTHING = 0.18

# Maximum horizontal movement between tracking samples.
# Prevents the crop from jumping too aggressively.
MAX_MOVE_RATIO = 0.06


# ============================================================
# MEDIAPIPE
# ============================================================

_mp_face_detection = None


def _load_face_detector():
    """
    Load MediaPipe Face Detection once.
    """

    global _mp_face_detection

    if _mp_face_detection is not None:
        return _mp_face_detection

    print("🧠 Loading MediaPipe face detector...")

    _mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    print("✅ MediaPipe face detector loaded.")

    return _mp_face_detection


# ============================================================
# FACE DETECTION
# ============================================================

def _detect_faces(detector, frame):
    """
    Detect faces in one BGR OpenCV frame.

    Returns:
        [
            {
                "cx": center_x,
                "cy": center_y,
                "area": area,
                "score": confidence,
            },
            ...
        ]
    """

    if frame is None:
        return []

    h, w = frame.shape[:2]

    if w <= 0 or h <= 0:
        return []

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = detector.process(rgb)

    if not result.detections:
        return []

    faces = []

    for detection in result.detections:

        score = 0.0

        if detection.score:
            score = float(detection.score[0])

        if score < MIN_DETECTION_CONFIDENCE:
            continue

        bbox = detection.location_data.relative_bounding_box

        x = bbox.xmin * w
        y = bbox.ymin * h
        bw = bbox.width * w
        bh = bbox.height * h

        # Clamp bounding box
        x1 = max(0.0, min(float(w), x))
        y1 = max(0.0, min(float(h), y))

        x2 = max(0.0, min(float(w), x + bw))
        y2 = max(0.0, min(float(h), y + bh))

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        area = bw * bh

        faces.append(
            {
                "cx": cx,
                "cy": cy,
                "area": area,
                "score": score,
            }
        )

    return faces


# ============================================================
# FACE SELECTION
# ============================================================

def _choose_face(faces, previous_x, frame_width):
    """
    Select the most likely face to follow.

    If we already have a previous position:
        prefer the face closest to it.

    Otherwise:
        prefer the largest/high-confidence face.
    """

    if not faces:
        return None

    # --------------------------------------------------------
    # First detection
    # --------------------------------------------------------

    if previous_x is None:

        return max(
            faces,
            key=lambda face: (
                face["area"] * 0.75
                + face["score"] * frame_width * frame_width * 0.25
            ),
        )

    # --------------------------------------------------------
    # Tracking existing face
    # --------------------------------------------------------

    best_face = None
    best_score = float("-inf")

    max_area = max(face["area"] for face in faces)

    for face in faces:

        distance = abs(face["cx"] - previous_x)

        # Normalize distance
        distance_ratio = distance / max(frame_width, 1)

        # Prefer faces near previous position
        distance_score = max(0.0, 1.0 - distance_ratio)

        # Slight preference for larger faces
        area_score = face["area"] / max(max_area, 1.0)

        # Detection confidence
        confidence_score = face["score"]

        score = (
            distance_score * 0.60
            + area_score * 0.20
            + confidence_score * 0.20
        )

        if score > best_score:
            best_score = score
            best_face = face

    return best_face


# ============================================================
# FACE TRACKING
# ============================================================

def _track_face(video_path, start_time, end_time):
    """
    Analyze the clip and return face X positions.

    Returns:
        list of:
            (time, face_center_x)

    If a face disappears temporarily, the previous position
    is kept.

    If no face is ever detected, returns an empty list.
    """

    detector = _load_face_detector()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print("⚠️ Could not open video for face tracking.")
        return []

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"🎯 Tracking resolution: "
        f"{source_width}x{source_height}"
    )

    points = []

    previous_x = None
    detected_count = 0
    total_samples = 0

    duration = max(0.0, end_time - start_time)

    sample_time = 0.0

    while sample_time <= duration + 0.001:

        absolute_time = start_time + sample_time

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            absolute_time * 1000.0,
        )

        ok, frame = capture.read()

        if not ok or frame is None:
            sample_time += TRACK_INTERVAL
            continue

        total_samples += 1

        faces = _detect_faces(detector, frame)

        face = _choose_face(
            faces,
            previous_x,
            source_width,
        )

        if face is not None:

            current_x = float(face["cx"])

            # ------------------------------------------------
            # Smooth raw detection
            # ------------------------------------------------

            if previous_x is not None:

                current_x = (
                    previous_x * (1.0 - SMOOTHING)
                    + current_x * SMOOTHING
                )

                # ------------------------------------------------
                # Limit maximum movement
                # ------------------------------------------------

                max_move = source_width * MAX_MOVE_RATIO

                delta = current_x - previous_x

                if delta > max_move:
                    current_x = previous_x + max_move

                elif delta < -max_move:
                    current_x = previous_x - max_move

            previous_x = current_x

            detected_count += 1

            points.append(
                (
                    sample_time,
                    current_x,
                )
            )

        else:

            # ------------------------------------------------
            # Face temporarily disappeared.
            # Keep the last known position.
            # ------------------------------------------------

            if previous_x is not None:
                points.append(
                    (
                        sample_time,
                        previous_x,
                    )
                )

        sample_time += TRACK_INTERVAL

    capture.release()

    print(
        f"👤 Face detections: "
        f"{detected_count}/{total_samples}"
    )

    if not points:
        print("❌ No face detected in clip.")
        print("↩️ Using center crop.")

        return []

    print(
        f"✅ Face tracking points: {len(points)}"
    )

    return points


# ============================================================
# SMOOTH TRACKING
# ============================================================

def _smooth_tracking(points, crop_w, source_width):
    """
    Additional smoothing and crop-boundary correction.
    """

    if not points:
        return []

    max_x = max(
        0.0,
        float(source_width - crop_w),
    )

    result = []

    previous_crop_x = None

    for timestamp, face_x in points:

        # We want the face approximately in the middle
        target_crop_x = face_x - crop_w / 2.0

        # Keep crop inside source video
        target_crop_x = max(
            0.0,
            min(max_x, target_crop_x),
        )

        if previous_crop_x is not None:

            target_crop_x = (
                previous_crop_x * (1.0 - SMOOTHING)
                + target_crop_x * SMOOTHING
            )

            max_move = source_width * MAX_MOVE_RATIO

            delta = target_crop_x - previous_crop_x

            if delta > max_move:
                target_crop_x = previous_crop_x + max_move

            elif delta < -max_move:
                target_crop_x = previous_crop_x - max_move

        previous_crop_x = target_crop_x

        result.append(
            (
                timestamp,
                target_crop_x,
            )
        )

    return result


# ============================================================
# FFMPEG EXPRESSION
# ============================================================

def _build_crop_expression(points, crop_w, source_width):
    """
    Build an FFmpeg expression for dynamic X crop.

    Example:

        crop=405:720:
        if(
            lt(t,1.0),
            100,
            if(
                lt(t,1.25),
                120,
                ...
            )
        ):
        0

    The expression interpolates between tracking points
    instead of jumping from one point to another.
    """

    if not points:
        center_x = max(
            0,
            int(round((source_width - crop_w) / 2)),
        )

        return str(center_x)

    max_x = max(
        0.0,
        float(source_width - crop_w),
    )

    # --------------------------------------------------------
    # Clamp all points
    # --------------------------------------------------------

    clean_points = []

    for timestamp, crop_x in points:

        crop_x = max(
            0.0,
            min(max_x, float(crop_x)),
        )

        clean_points.append(
            (
                float(timestamp),
                crop_x,
            )
        )

    if len(clean_points) == 1:
        return str(int(round(clean_points[0][1])))

    # --------------------------------------------------------
    # Build piecewise linear expression.
    #
    # We build it backwards because FFmpeg expressions are
    # easier to generate this way.
    # --------------------------------------------------------

    expression = str(
        int(round(clean_points[-1][1]))
    )

    for i in range(len(clean_points) - 2, -1, -1):

        t0, x0 = clean_points[i]
        t1, x1 = clean_points[i + 1]

        dt = max(0.001, t1 - t0)

        slope = (x1 - x0) / dt

        segment_expression = (
            f"({x0:.4f}+"
            f"({slope:.6f})*(t-{t0:.4f}))"
        )

        expression = (
            f"if(lt(t,{t1:.4f}),"
            f"{segment_expression},"
            f"{expression})"
        )

    return expression


# ============================================================
# CREATE FACE CROP
# ============================================================

def _create_face_crop(
    video_path,
    start_time,
    end_time,
):
    """
    Create a dynamic 9:16 crop based on face position.

    Source example:
        1280x720

    Crop:
        405x720

    Output:
        1080x1920
    """

    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]

    result = subprocess.run(
        probe_cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    resolution = result.stdout.strip()

    source_width, source_height = map(
        int,
        resolution.split("x"),
    )

    print(
        f"📐 Source resolution: "
        f"{source_width}x{source_height}"
    )

    # --------------------------------------------------------
    # 9:16 crop
    # --------------------------------------------------------

    crop_h = source_height

    crop_w = int(
        round(crop_h * 9 / 16)
    )

    # Safety
    crop_w = min(
        crop_w,
        source_width,
    )

    print(
        f"📐 Crop area: "
        f"{crop_w}x{crop_h}"
    )

    # --------------------------------------------------------
    # Track face
    # --------------------------------------------------------

    tracking_points = _track_face(
        video_path,
        start_time,
        end_time,
    )

    # --------------------------------------------------------
    # No face
    # --------------------------------------------------------

    if not tracking_points:

        center_x = max(
            0,
            int(
                round(
                    (source_width - crop_w) / 2
                )
            ),
        )

        print(
            f"↩️ Center crop X: {center_x}"
        )

        return {
            "crop_w": crop_w,
            "crop_h": crop_h,
            "x_expression": str(center_x),
        }

    # --------------------------------------------------------
    # Convert face centers to crop positions
    # --------------------------------------------------------

    crop_points = _smooth_tracking(
        tracking_points,
        crop_w,
        source_width,
    )

    # --------------------------------------------------------
    # Build FFmpeg expression
    # --------------------------------------------------------

    x_expression = _build_crop_expression(
        crop_points,
        crop_w,
        source_width,
    )

    print(
        "🎯 Dynamic face crop enabled."
    )

    print(
        f"🎯 Tracking points: "
        f"{len(crop_points)}"
    )

    return {
        "crop_w": crop_w,
        "crop_h": crop_h,
        "x_expression": x_expression,
    }


# ============================================================
# RENDER ONE CLIP
# ============================================================

def render_clip(
    source_video,
    output_path,
    start_time,
    end_time,
):
    """
    Render one vertical Shorts clip.

    Preserves:
        1080x1920
        H.264
        CRF 20
        AAC 192k
        original audio
        faststart
    """

    source_video = Path(source_video)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print("🎬 Rendering clip")
    print(
        f"⏱️ Start: {start_time:.2f}s"
    )
    print(
        f"⏱️ End:   {end_time:.2f}s"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Create dynamic face crop
    # --------------------------------------------------------

    crop = _create_face_crop(
        source_video,
        start_time,
        end_time,
    )

    crop_w = crop["crop_w"]
    crop_h = crop["crop_h"]
    x_expression = crop["x_expression"]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do not quote the whole crop expression as one argument.
    # FFmpeg receives it as part of the filter string.
    # --------------------------------------------------------

    crop_filter = (
        f"crop="
        f"{crop_w}:"
        f"{crop_h}:"
        f"{x_expression}:"
        f"0,"
        f"scale="
        f"{WIDTH}:"
        f"{HEIGHT}:"
        f"flags=lanczos"
    )

    cmd = [
        "ffmpeg",
        "-y",

        "-ss",
        str(start_time),

        "-i",
        str(source_video),

        "-t",
        str(end_time - start_time),

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

        "-movflags",
        "+faststart",

        str(output_path),
    ]

    print("🎥 Running FFmpeg...")

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if process.returncode != 0:

        print(
            "❌ FFmpeg render failed:"
        )

        print(
            process.stdout[-8000:]
        )

        raise RuntimeError(
            "FFmpeg render failed"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"Output file was not created: "
            f"{output_path}"
        )

    size_mb = (
        output_path.stat().st_size
        / 1024
        / 1024
    )

    print(
        f"✅ Clip rendered: "
        f"{output_path}"
    )

    print(
        f"📦 Size: {size_mb:.2f} MB"
    )

    return output_path


# ============================================================
# RENDER ALL CLIPS
# ============================================================

def render_clips(
    source_video,
    clips,
    output_dir,
):
    """
    Render all selected clips.

    Expected clips format:

        [
            {
                "start": 100.0,
                "end": 150.0,
            },
            ...
        ]

    Also supports:

        {
            "start_time": ...,
            "end_time": ...
        }
    """

    source_video = Path(source_video)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered = []

    for index, clip in enumerate(
        clips,
        start=1,
    ):

        # ----------------------------------------------------
        # Support both naming formats
        # ----------------------------------------------------

        start_time = clip.get(
            "start",
            clip.get("start_time"),
        )

        end_time = clip.get(
            "end",
            clip.get("end_time"),
        )

        if start_time is None or end_time is None:
            print(
                f"⚠️ Invalid clip #{index}: "
                f"{clip}"
            )
            continue

        start_time = float(start_time)
        end_time = float(end_time)

        if end_time <= start_time:
            print(
                f"⚠️ Invalid clip duration "
                f"#{index}: "
                f"{start_time} -> {end_time}"
            )
            continue

        output_path = (
            output_dir
            / f"clip_{index:02d}.mp4"
        )

        rendered_path = render_clip(
            source_video=source_video,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
        )

        rendered.append(
            rendered_path
        )

    print()
    print("=" * 60)
    print(
        f"✅ Rendered clips: "
        f"{len(rendered)}"
    )
    print("=" * 60)

    return rendered
