from PIL import Image, ImageDraw, ImageFont, ImageFilter

import numpy as np

from moviepy import ImageClip


FONT_PATH = "assets/fonts/Montserrat-ExtraBold.ttf"


WIDTH = 1080
HEIGHT = 1920


FONT_SIZE = 70

POSITION_Y = 0.65

FADE_TIME = 0.12


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

        fill=(
            0,
            0,
            0,
            200
        )
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

        fill=(
            255,
            255,
            255,
            255
        )
    )


    return np.array(
        img
    )


# ==========================
# CREATE SUBTITLES
# ==========================

def create_subtitle(words):

    clips = []


    for item in words:

        text = item["word"].upper()

        start = item["start"]

        end = item["end"]

        duration = end - start


        if duration <= 0:

            continue


        img = create_text_image(
            text
        )


        clip = ImageClip(
            img,
            duration=duration
        )


        clip = clip.with_start(
            start
        )


        # ==========================
        # OPACITY ANIMATION
        # ==========================

        def opacity(t):

            if t < FADE_TIME:

                return t / FADE_TIME


            if t > duration - FADE_TIME:

                return (
                    duration - t
                ) / FADE_TIME


            return 1.0


        clip = clip.with_opacity(
            opacity
        )


        clips.append(
            clip
        )


    return clips
