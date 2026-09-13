import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


WIDTH = 1080
HEIGHT = 1920

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def extract_frame(video_path: Path, output_path: Path, timestamp: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed to extract thumbnail frame:\n{result.stderr}"
        )


def crop_to_vertical(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")

    target_ratio = WIDTH / HEIGHT
    image_ratio = image.width / image.height

    if image_ratio > target_ratio:
        new_width = int(image.height * target_ratio)
        left = (image.width - new_width) // 2
        image = image.crop(
            (left, 0, left + new_width, image.height)
        )
    else:
        new_height = int(image.width / target_ratio)
        top = (image.height - new_height) // 2
        image = image.crop(
            (0, top, image.width, top + new_height)
        )

    return image.resize(
        (WIDTH, HEIGHT),
        Image.Resampling.LANCZOS,
    )


def shorten_title(title: str) -> str:
    title = title.strip()

    replacements = {
        "Как Стивен Хокинг упал со сцены прямо во время лекции":
            "ХОКИНГ УПАЛ СО СЦЕНЫ",
        "Как физика разрешает путешествия во времени?":
            "ПУТЕШЕСТВИЯ ВО ВРЕМЕНИ?",
        "Ученый случайно изобрел машину времени?":
            "МАШИНА ВРЕМЕНИ ИЗОБРЕТЕНА?",
    }

    if title in replacements:
        return replacements[title]

    words = title.replace("?", "").replace("!", "").split()

    if len(words) <= 5:
        return " ".join(words).upper()

    return " ".join(words[:5]).upper()


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()

        bbox = draw.textbbox(
            (0, 0),
            test,
            font=font,
            stroke_width=2,
        )

        if bbox[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def create_thumbnail(
    video_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> Path:

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    title = str(metadata.get("title", "")).strip()

    if not title:
        raise RuntimeError(
            f"Metadata has no title: {metadata_path}"
        )

    # Берём кадр примерно из первой трети ролика.
    # Это обычно лучше, чем самый первый кадр.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    duration = float(probe.stdout.strip())
    timestamp = max(0.5, duration * 0.35)

    temp_frame = output_path.with_suffix(".frame.jpg")

    try:
        extract_frame(
            video_path,
            temp_frame,
            timestamp,
        )

        image = Image.open(temp_frame)
        image = crop_to_vertical(image)

        # Немного затемняем фон,
        # чтобы текст хорошо читался.
        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0),
        )

        overlay_draw = ImageDraw.Draw(overlay)

        for y in range(HEIGHT):
            alpha = int(170 * (y / HEIGHT))
            overlay_draw.line(
                [(0, y), (WIDTH, y)],
                fill=(0, 0, 0, alpha),
            )

        image = Image.alpha_composite(
            image.convert("RGBA"),
            overlay,
        )

        draw = ImageDraw.Draw(image)

        font_size = 100

        while font_size >= 60:
            font = ImageFont.truetype(
                FONT_PATH,
                font_size,
            )

            lines = wrap_text(
                draw,
                shorten_title(title),
                font,
                WIDTH - 120,
            )

            if len(lines) <= 3:
                break

            font_size -= 8

        # Основной текст.
        line_height = font_size + 20
        total_height = len(lines) * line_height

        start_y = HEIGHT - total_height - 260

        for line in lines:
            bbox = draw.textbbox(
                (0, 0),
                line,
                font=font,
                stroke_width=8,
            )

            text_width = bbox[2] - bbox[0]
            x = (WIDTH - text_width) // 2

            draw.text(
                (x, start_y),
                line,
                font=font,
                fill="white",
                stroke_width=8,
                stroke_fill="black",
            )

            start_y += line_height

        # Небольшой бейдж сверху.
        badge_font = ImageFont.truetype(
            FONT_PATH,
            48,
        )

        badge_text = "SHORTS"

        badge_bbox = draw.textbbox(
            (0, 0),
            badge_text,
            font=badge_font,
        )

        badge_width = badge_bbox[2] - badge_bbox[0] + 60
        badge_height = badge_bbox[3] - badge_bbox[1] + 30

        badge_x = 50
        badge_y = 60

        draw.rounded_rectangle(
            (
                badge_x,
                badge_y,
                badge_x + badge_width,
                badge_y + badge_height,
            ),
            radius=20,
            fill=(220, 30, 30, 255),
        )

        draw.text(
            (
                badge_x + 30,
                badge_y + 12,
            ),
            badge_text,
            font=badge_font,
            fill="white",
        )

        image = image.convert("RGB")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            output_path,
            "JPEG",
            quality=92,
            optimize=True,
        )

    finally:
        if temp_frame.exists():
            temp_frame.unlink()

    return output_path
