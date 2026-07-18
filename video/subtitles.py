from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

from moviepy import ImageClip


FONT_PATH = "assets/fonts/Montserrat-ExtraBold.ttf"


WIDTH = 1080
HEIGHT = 1920


FONT_SIZE = 70

POSITION_Y = 0.65


# ==========================
# ANIMATION SETTINGS
# ==========================

# Длительность появления
ANIMATION_IN = 0.10

# Длительность исчезновения
ANIMATION_OUT = 0.10

# Начальный масштаб слова
START_SCALE = 0.88

# Финальный масштаб
END_SCALE = 1.0


# ==========================
# CREATE TEXT IMAGE
# ==========================

def create_text_image(text):

    img = Image.new(
        "RGBA",
        (WIDTH, HEIGHT),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(
        FONT_PATH,
        FONT_SIZE
    )

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (WIDTH - text_width) / 2

    y = HEIGHT * POSITION_Y


    # ==========================
    # SHADOW
    # ==========================

    shadow = Image.new(
        "RGBA",
        (WIDTH, HEIGHT),
        (0, 0, 0, 0)
    )

    shadow_draw = ImageDraw.Draw(
        shadow
    )

    shadow_draw.text(
        (
            x + 5,
            y + 5
        ),
        text,
        font=font,
        fill=(0, 0, 0, 200)
    )

    shadow = shadow.filter(
        ImageFilter.GaussianBlur(5)
    )

    img.alpha_composite(
        shadow
    )


    # ==========================
    # MAIN TEXT
    # ==========================

    draw.text(
        (
            x,
            y
        ),
        text,
        font=font,
        fill=(255, 255, 255, 255)
    )


    return np.array(img)


# ==========================
# ANIMATED SUBTITLE
# ==========================

def create_animated_word(
    text,
    start,
    end
):

    duration = end - start


    if duration <= 0:

        return None


    image = create_text_image(
        text
    )


    clip = ImageClip(
        image,
        duration=duration
    )


    # ==========================
    # TIMING
    # ==========================

    fade_in_end = min(
        ANIMATION_IN,
        duration / 2
    )


    fade_out_start = max(
        duration - ANIMATION_OUT,
        fade_in_end
    )


    # ==========================
    # OPACITY ANIMATION
    # ==========================

    def opacity_animation(t):

        if t < fade_in_end:

            progress = t / fade_in_end

            return progress


        if t > fade_out_start:

            progress = (
                duration - t
            ) / ANIMATION_OUT

            return max(
                0,
                progress
            )


        return 1


    clip = clip.with_opacity(
        opacity_animation
    )


    # ==========================
    # SCALE ANIMATION
    # ==========================

    def scale_animation(t):

        if t < fade_in_end:

            progress = t / fade_in_end

            return (
                START_SCALE
                +
                (
                    END_SCALE
                    -
                    START_SCALE
                )
                *
                progress
            )


        return END_SCALE


    clip = clip.resized(
        scale_animation
    )


    # ==========================
    # TIMING
    # ==========================

    clip = clip.with_start(
        start
    )


    return clip


# ==========================
# CREATE SUBTITLES
# ==========================

def create_subtitle(words):

    clips = []


    for item in words:


        text = item["word"].upper()


        start = item["start"]


        end = item["end"]


        clip = create_animated_word(

            text,

            start,

            end

        )


        if clip is not None:

            clips.append(
                clip
            )


    print(
        f"💬 Создано субтитров: {len(clips)}"
    )


    return clips
