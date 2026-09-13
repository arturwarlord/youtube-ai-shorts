import subprocess
from pathlib import Path


def _run_ffmpeg(command):
    print("💬 Running FFmpeg subtitles...")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"FFmpeg subtitles failed with exit code {result.returncode}"
        )

    return result


def _escape_ass_text(text):
    """
    Escape text for ASS subtitle format.
    """
    text = str(text)

    text = text.replace("\\", r"\\")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")

    return text


def create_ass_subtitles(words, start, end, output_path):
    """
    Creates an ASS subtitle file for one selected clip.

    words:
        Whisper word timestamps.

    start/end:
        Original source-video timestamps.

    output_path:
        Path to .ass file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = []

    for word in words:
        word_start = float(word.get("start", 0))
        word_end = float(word.get("end", word_start))

        if word_end <= start:
            continue

        if word_start >= end:
            break

        text = str(word.get("word", "")).strip()

        if not text:
            continue

        clip_start = max(word_start, start) - start
        clip_end = min(word_end, end) - start

        if clip_end <= clip_start:
            continue

        selected.append(
            {
                "text": text,
                "start": clip_start,
                "end": clip_end,
            }
        )

    if not selected:
        raise RuntimeError(
            f"No subtitle words found for clip "
            f"{start:.2f} -> {end:.2f}"
        )

    def ass_time(seconds):
        seconds = max(0, float(seconds))

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int(round((seconds - int(seconds)) * 100))

        if centiseconds >= 100:
            centiseconds = 0
            secs += 1

        if secs >= 60:
            secs = 0
            minutes += 1

        if minutes >= 60:
            minutes = 0
            hours += 1

        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    # ---------------------------------------------------------
    # Group words into short readable phrases.
    # ---------------------------------------------------------

    groups = []

    current = []

    for item in selected:
        current.append(item)

        current_text = " ".join(
            x["text"]
            for x in current
        )

        # 2–4 words per subtitle block.
        if (
            len(current) >= 4
            or len(current_text) >= 28
        ):
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    # ---------------------------------------------------------
    # ASS document.
    # ---------------------------------------------------------

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, "
        "SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Shorts,Arial,82,"
        "&H00FFFFFF,"
        "&H00FFFFFF,"
        "&H00000000,"
        "&H80000000,"
        "1,0,0,0,"
        "100,100,0,0,"
        "1,5,2,"
        "2,60,60,360,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text",
    ]

    for group in groups:
        group_start = group[0]["start"]
        group_end = group[-1]["end"]

        text = " ".join(
            _escape_ass_text(x["text"])
            for x in group
        )

        lines.append(
            f"Dialogue: 0,"
            f"{ass_time(group_start)},"
            f"{ass_time(group_end)},"
            f"Shorts,,0,0,0,,"
            f"{text}"
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"✅ Subtitles created: {output_path}"
    )

    return str(output_path)


def burn_subtitles(
    video_path,
    subtitle_path,
    output_path,
):
    """
    Burns ASS subtitles directly into the video.
    """

    video_path = Path(video_path)
    subtitle_path = Path(subtitle_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not subtitle_path.exists():
        raise FileNotFoundError(
            f"Subtitle file not found: {subtitle_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"ass={subtitle_path}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    _run_ffmpeg(command)

    if not output_path.exists():
        raise RuntimeError(
            f"Subtitle video was not created: {output_path}"
        )

    size_mb = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    print("✅ Subtitles burned into video")
    print(f"📁 {output_path}")
    print(f"💾 {size_mb:.2f} MB")

    return str(output_path)
