import subprocess
from pathlib import Path

import cv2
import mediapipe as mp


WIDTH = 1080
HEIGHT = 1920

TRACK_INTERVAL = 0.25
MIN_DETECTION_CONFIDENCE = 0.45

SMOOTHING = 0.18
MAX_MOVE_RATIO = 0.06


# ============================================================
# MediaPipe
# ============================================================

def _load_face_detector():
    """
    Create MediaPipe face detector.

    model_selection=1 is better for faces that are not extremely
    close to the camera.
    """
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )


def _detect_faces(detector, frame):
    """
    Detect faces in a BGR OpenCV frame.

    Returns:
        [
            {
                "x": center_x,
                "y": center_y,
                "area": area,
                "score": confidence,
            }
        ]
    """
    if frame is None or frame.size == 0:
        return []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector.process(rgb)

    if not results.detections:
        return []

    height, width = frame.shape[:2]

    faces = []

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x = max(0.0, bbox.xmin)
        y = max(0.0, bbox.ymin)

        w = max(0.0, bbox.width)
        h = max(0.0, bbox.height)

        center_x = (x + w / 2.0) * width
        center_y = (y + h / 2.0) * height

        area = (w * width) * (h * height)

        score = 0.0

        if detection.score:
            score = float(detection.score[0])

        faces.append(
            {
                "x": center_x,
                "y": center_y,
                "area": area,
                "score": score,
            }
        )

    return faces


def _choose_face(faces, previous_x=None):
    """
    Choose the most suitable face.

    If we already have a previous X position,
    prefer the face closest to it.

    Otherwise prefer the largest / most confident face.
    """
    if not faces:
        return None

    if previous_x is not None:
        def distance_score(face):
            distance = abs(face["x"] - previous_x)

            confidence_bonus = face["score"] * 100.0
            area_bonus = min(face["area"] / 10000.0, 100.0)

            return distance - confidence_bonus - area_bonus

        return min(faces, key=distance_score)

    def initial_score(face):
        return (
            face["area"] * max(face["score"], 0.1)
        )

    return max(faces, key=initial_score)


# ============================================================
# Face tracking
# ============================================================

def _track_face(
    video_path,
    clip_start,
    clip_duration,
    source_width,
    source_height,
):
    """
    Track the face through the selected clip.

    Returns:
        [
            (time_in_clip, face_center_x)
        ]
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video for face tracking: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    video_duration = total_frames / fps

    start_time = max(0.0, float(clip_start))
    end_time = min(
        video_duration,
        start_time + max(0.0, float(clip_duration)),
    )

    if end_time <= start_time:
        cap.release()
        return []

    detector = _load_face_detector()

    points = []

    previous_x = None
    smoothed_x = None

    sample_time = 0.0

    while sample_time <= (end_time - start_time) + 0.001:
        absolute_time = start_time + sample_time

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            absolute_time * 1000.0,
        )

        success, frame = cap.read()

        if not success or frame is None:
            sample_time += TRACK_INTERVAL
            continue

        faces = _detect_faces(
            detector,
            frame,
        )

        face = _choose_face(
            faces,
            previous_x,
        )

        if face is not None:
            detected_x = float(face["x"])

            if smoothed_x is None:
                smoothed_x = detected_x
            else:
                smoothed_x = (
                    smoothed_x * (1.0 - SMOOTHING)
                    + detected_x * SMOOTHING
                )

            previous_x = smoothed_x

            points.append(
                (
                    float(sample_time),
                    float(smoothed_x),
                )
            )

        sample_time += TRACK_INTERVAL

    detector.close()
    cap.release()

    return points


# ============================================================
# Crop smoothing
# ============================================================

def _smooth_tracking(
    points,
    crop_w,
    source_width,
):
    """
    Convert face-center positions into crop X positions.

    Keeps the crop inside the source frame.
    """

    if not points:
        center_x = max(
            0,
            int(round((source_width - crop_w) / 2)),
        )

        return [
            (
                0.0,
                float(center_x),
            )
        ]

    max_x = max(
        0.0,
        float(source_width - crop_w),
    )

    result = []

    previous_crop_x = None

    max_move = max(
        1.0,
        float(source_width) * MAX_MOVE_RATIO,
    )

    for timestamp, face_x in points:

        target_x = float(face_x) - (crop_w / 2.0)

        target_x = max(
            0.0,
            min(max_x, target_x),
        )

        if previous_crop_x is None:
            crop_x = target_x
        else:
            delta = target_x - previous_crop_x

            if delta > max_move:
                delta = max_move
            elif delta < -max_move:
                delta = -max_move

            crop_x = previous_crop_x + delta

        crop_x = max(
            0.0,
            min(max_x, crop_x),
        )

        result.append(
            (
                float(timestamp),
                float(crop_x),
            )
        )

        previous_crop_x = crop_x

    return result


# ============================================================
# FFmpeg crop expression
# ============================================================

def _build_crop_expression(
    points,
    crop_w,
    source_width,
):
    """
    Build a compact FFmpeg expression for dynamic X crop.

    MediaPipe may produce hundreds of tracking points.
    FFmpeg does not need all of them.

    We reduce the points to a maximum of 20 keyframes,
    which keeps the FFmpeg expression small and reliable.
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

    clean_points = []

    for timestamp, crop_x in points:

        timestamp = float(timestamp)

        crop_x = max(
            0.0,
            min(
                max_x,
                float(crop_x),
            ),
        )

        clean_points.append(
            (
                timestamp,
                crop_x,
            )
        )

    if not clean_points:
        center_x = max(
            0,
            int(round((source_width - crop_w) / 2)),
        )

        return str(center_x)

    if len(clean_points) == 1:
        return str(
            int(
                round(
                    clean_points[0][1]
                )
            )
        )

    # --------------------------------------------------------
    # Important:
    #
    # MediaPipe tracking is done every 0.25 sec.
    # A long video can therefore generate hundreds of points.
    #
    # FFmpeg's expression evaluator does not handle a gigantic
    # nested if() expression reliably.
    #
    # Keep only 20 representative points.
    # --------------------------------------------------------

    MAX_EXPRESSION_POINTS = 20

    if len(clean_points) > MAX_EXPRESSION_POINTS:

        reduced = []

        last_index = len(clean_points) - 1

        for i in range(MAX_EXPRESSION_POINTS):

            position = (
                i
                / (MAX_EXPRESSION_POINTS - 1)
            )

            index = int(
                round(
                    position * last_index
                )
            )

            reduced.append(
                clean_points[index]
            )

        clean_points = reduced

    # --------------------------------------------------------
    # Remove duplicate timestamps.
    # --------------------------------------------------------

    unique_points = []

    previous_timestamp = None

    for timestamp, crop_x in clean_points:

        if (
            previous_timestamp is not None
            and abs(
                timestamp - previous_timestamp
            ) < 0.0001
        ):
            continue

        unique_points.append(
            (
                timestamp,
                crop_x,
            )
        )

        previous_timestamp = timestamp

    clean_points = unique_points

    if len(clean_points) == 1:
        return str(
            int(
                round(
                    clean_points[0][1]
                )
            )
        )

    # --------------------------------------------------------
    # Build piecewise-linear expression.
    #
    # Example:
    #
    # if(
    #   lt(t,1.0),
    #   ...,
    #   if(
    #      lt(t,2.0),
    #      ...,
    #      ...
    #   )
    # )
    # --------------------------------------------------------

    expression = (
        f"{clean_points[-1][1]:.2f}"
    )

    for i in range(
        len(clean_points) - 2,
        -1,
        -1,
    ):

        t0, x0 = clean_points[i]
        t1, x1 = clean_points[i + 1]

        dt = max(
            0.001,
            t1 - t0,
        )

        slope = (
            (x1 - x0)
            / dt
        )

        segment_expression = (
            f"({x0:.2f}+"
            f"({slope:.4f})*"
            f"(t-{t0:.2f}))"
        )

        expression = (
            f"if(lt(t,{t1:.2f}),"
            f"{segment_expression},"
            f"{expression})"
        )

    return expression


def _escape_filter_expression(expression):
    """
    Escape commas inside FFmpeg expressions.

    crop=...:if(...,...,...):...
    needs commas escaped when passed as a filter string.
    """

    return expression.replace(
        ",",
        r"\,",
    )


# ============================================================
# Source dimensions
# ============================================================

def _get_video_dimensions(video_path):
    """
    Read source video dimensions with OpenCV.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video dimensions: "
            f"{width}x{height}"
        )

    return width, height


# ============================================================
# Create face crop
# ============================================================

def _create_face_crop(
    video_path,
    clip_start,
    clip_duration,
):
    """
    Calculate the 9:16 crop and generate the dynamic
    face-following X expression.
    """

    source_width, source_height = (
        _get_video_dimensions(video_path)
    )

    # --------------------------------------------------------
    # 9:16 crop
    # --------------------------------------------------------

    target_ratio = (
        WIDTH / HEIGHT
    )

    source_ratio = (
        source_width / source_height
    )

    if source_ratio > target_ratio:

        # Landscape / wider source.
        crop_h = source_height

        crop_w = int(
            round(
                crop_h * target_ratio
            )
        )

    else:

        # Already portrait or close to portrait.
        crop_w = source_width

        crop_h = int(
            round(
                crop_w / target_ratio
            )
        )

        if crop_h > source_height:
            crop_h = source_height

            crop_w = int(
                round(
                    crop_h * target_ratio
                )
            )

    crop_w = max(
        2,
        min(
            crop_w,
            source_width,
        ),
    )

    crop_h = max(
        2,
        min(
            crop_h,
            source_height,
        ),
    )

    print(
        f"📐 Source: "
        f"{source_width}x{source_height}"
    )

    print(
        f"📐 Crop: "
        f"{crop_w}x{crop_h}"
    )

    # --------------------------------------------------------
    # Track face.
    # --------------------------------------------------------

    print(
        "👤 Tracking face with MediaPipe..."
    )

    tracking_points = _track_face(
        video_path=video_path,
        clip_start=clip_start,
        clip_duration=clip_duration,
        source_width=source_width,
        source_height=source_height,
    )

    print(
        f"👤 Face tracking points: "
        f"{len(tracking_points)}"
    )

    # --------------------------------------------------------
    # If face wasn't detected, use center crop.
    # --------------------------------------------------------

    if not tracking_points:

        center_x = max(
            0,
            int(
                round(
                    (source_width - crop_w)
                    / 2
                )
            ),
        )

        print(
            "⚠️ Face not detected. "
            "Using center crop."
        )

        return (
            crop_w,
            crop_h,
            str(center_x),
        )

    # --------------------------------------------------------
    # Convert face centers into crop positions.
    # --------------------------------------------------------

    crop_points = _smooth_tracking(
        points=tracking_points,
        crop_w=crop_w,
        source_width=source_width,
    )

    # --------------------------------------------------------
    # Build compact FFmpeg expression.
    # --------------------------------------------------------

    x_expression = _build_crop_expression(
        points=crop_points,
        crop_w=crop_w,
        source_width=source_width,
    )

    print(
        f"🎯 Crop X expression: "
        f"{x_expression}"
    )

    return (
        crop_w,
        crop_h,
        x_expression,
    )


# ============================================================
# Render single clip
# ============================================================

def render_clip(
    source_path,
    output_path,
    start_time,
    end_time,
):
    """
    Render one vertical 1080x1920 Short.

    Uses:
        - MediaPipe face tracking
        - dynamic horizontal crop
        - 9:16 framing
        - Lanczos scaling
        - H.264
        - AAC
    """

    source_path = Path(
        source_path
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = float(
        start_time
    )

    end_time = float(
        end_time
    )

    duration = (
        end_time
        - start_time
    )

    if duration <= 0:
        raise ValueError(
            f"Invalid clip duration: "
            f"{start_time} -> {end_time}"
        )

    print()
    print(
        "🎬 Rendering clip"
    )

    print(
        f"   Start: {start_time:.2f}s"
    )

    print(
        f"   End:   {end_time:.2f}s"
    )

    print(
        f"   Duration: {duration:.2f}s"
    )

    # --------------------------------------------------------
    # Create dynamic face-following crop.
    # --------------------------------------------------------

    crop_w, crop_h, x_expression = (
        _create_face_crop(
            video_path=source_path,
            clip_start=start_time,
            clip_duration=duration,
        )
    )

    # --------------------------------------------------------
    # Escape commas for FFmpeg filter parser.
    # --------------------------------------------------------

    escaped_x_expression = (
        _escape_filter_expression(
            x_expression
        )
    )

    crop_filter = (
        f"crop="
        f"{crop_w}:"
        f"{crop_h}:"
        f"{escaped_x_expression}:0,"
        f"scale="
        f"{WIDTH}:"
        f"{HEIGHT}:"
        f"flags=lanczos"
    )

    print(
        f"🎯 Crop filter: "
        f"{crop_filter}"
    )

    # --------------------------------------------------------
    # FFmpeg command.
    #
    # Keep -ss before -i for fast seeking.
    # --------------------------------------------------------

    command = [
        "ffmpeg",
        "-y",

        "-ss",
        str(start_time),

        "-i",
        str(source_path),

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

        "-movflags",
        "+faststart",

        str(output_path),
    ]

    print()
    print(
        "🚀 Running FFmpeg..."
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        print()
        print(
            "❌ FFmpeg failed:"
        )

        print(
            result.stderr
        )

        raise RuntimeError(
            "FFmpeg render failed"
        )

    if (
        not output_path.exists()
        or output_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            f"FFmpeg completed but output "
            f"file is missing or empty: "
            f"{output_path}"
        )

    print()
    print(
        f"✅ Rendered: "
        f"{output_path}"
    )

    print(
        f"📦 Size: "
        f"{output_path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    return output_path


# ============================================================
# Render multiple clips
# ============================================================

def render_clips(
    source_path,
    clips,
    output_dir,
):
    """
    Render multiple clips.

    Supports clip dictionaries in both forms:

        {
            "start": 10,
            "end": 40
        }

    and:

        {
            "start_time": 10,
            "end_time": 40
        }
    """

    source_path = Path(
        source_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered_files = []

    if not clips:
        print(
            "⚠️ No clips to render."
        )

        return rendered_files

    print()
    print(
        f"🎞️ Clips to render: "
        f"{len(clips)}"
    )

    for index, clip in enumerate(
        clips,
        start=1,
    ):

        if "start" in clip:
            start_time = float(
                clip["start"]
            )

        elif "start_time" in clip:
            start_time = float(
                clip["start_time"]
            )

        else:
            raise ValueError(
                f"Clip {index} has no "
                f"'start' or 'start_time'"
            )

        if "end" in clip:
            end_time = float(
                clip["end"]
            )

        elif "end_time" in clip:
            end_time = float(
                clip["end_time"]
            )

        else:
            raise ValueError(
                f"Clip {index} has no "
                f"'end' or 'end_time'"
            )

        output_path = (
            output_dir
            / f"clip_{index:02d}.mp4"
        )

        print()
        print(
            "========================================"
        )

        print(
            f"🎬 Clip {index}/{len(clips)}"
        )

        print(
            "========================================"
        )

        rendered = render_clip(
            source_path=source_path,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
        )

        rendered_files.append(
            rendered
        )

    print()
    print(
        "========================================"
    )

    print(
        f"✅ Rendered "
        f"{len(rendered_files)} clips"
    )

    print(
        "========================================"
    )

    return rendered_files
