import os

from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip


# ==========================================
# CONFIG
# ==========================================

OUTPUT_DIR = "output"

THUMBNAIL_PATH = os.path.join(
    OUTPUT_DIR,
    "thumbnail.jpg"
)

WIDTH = 1080
HEIGHT = 1920


# ==========================================
# FONT
# ==========================================

FONT_PATH = (
    "assets/fonts/"
    "Montserrat-ExtraBold.ttf"
)


# ==========================================
# CREATE THUMBNAIL
# ==========================================

def create_thumbnail(
    video_path,
    title
):

    print(
        "\n🖼️ Создание автоматической обложки..."
    )


    # ======================================
    # CHECK VIDEO
    # ======================================

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Видео не найдено: {video_path}"
        )


    # ======================================
    # OPEN VIDEO
    # ======================================

    video = VideoFileClip(
        video_path
    )


    # Берём кадр примерно из первой трети
    duration = video.duration

    frame_time = min(
        duration * 0.35,
        duration - 0.1
    )


    frame = video.get_frame(
        frame_time
    )


    video.close()


    # ======================================
    # PIL IMAGE
    # ======================================

    image = Image.fromarray(
        frame
    )


    image = image.resize(
        (
            WIDTH,
            HEIGHT
        )
    )


    # ======================================
    # DARK OVERLAY
    # ======================================

    overlay = Image.new(
        "RGBA",
        image.size,
        (
            0,
            0,
            0,
            90
        )
    )


    image = Image.alpha_composite(
        image.convert("RGBA"),
        overlay
    )


    draw = ImageDraw.Draw(
        image
    )


    # ======================================
    # TITLE
    # ======================================

    title = title.upper()


    # Убираем слишком длинный заголовок
    if len(title) > 60:

        title = title[:57] + "..."


    # ======================================
    # FONT SIZE
    # ======================================

    font_size = 92


    font = ImageFont.truetype(
        FONT_PATH,
        font_size
    )


    # ======================================
    # WRAP TITLE
    # ======================================

    words = title.split()

    lines = []

    current_line = ""


    for word in words:

        test_line = (
            current_line
            + " "
            + word
        ).strip()


        bbox = draw.textbbox(
            (
                0,
                0
            ),
            test_line,
            font=font
        )


        text_width = (
            bbox[2] - bbox[0]
        )


        if text_width <= 900:

            current_line = test_line

        else:

            if current_line:

                lines.append(
                    current_line
                )

            current_line = word


    if current_line:

        lines.append(
            current_line
        )


    # ======================================
    # TEXT POSITION
    # ======================================

    line_height = 115

    total_height = (
        len(lines)
        * line_height
    )


    start_y = (
        HEIGHT
        - total_height
    ) // 2


    # ======================================
    # DRAW TEXT
    # ======================================

    for index, line in enumerate(lines):

        bbox = draw.textbbox(
            (
                0,
                0
            ),
            line,
            font=font
        )


        text_width = (
            bbox[2] - bbox[0]
        )


        x = (
            WIDTH
            - text_width
        ) // 2


        y = (
            start_y
            + index
            * line_height
        )


        # Shadow
        draw.text(
            (
                x + 6,
                y + 6
            ),
            line,
            font=font,
            fill=(
                0,
                0,
                0,
                180
            )
        )


        # Main text
        draw.text(
            (
                x,
                y
            ),
            line,
            font=font,
            fill=(
                255,
                255,
                255,
                255
            )
        )


    # ======================================
    # SAVE
    # ======================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    image.convert(
        "RGB"
    ).save(
        THUMBNAIL_PATH,
        "JPEG",
        quality=95
    )


    print(
        f"✅ Обложка создана: "
        f"{THUMBNAIL_PATH}"
    )


    return THUMBNAIL_PATH
