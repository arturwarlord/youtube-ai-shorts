from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import textwrap

from moviepy import ImageClip


FONT_PATH = "assets/fonts/Montserrat-ExtraBold.ttf"


WIDTH = 1080
HEIGHT = 1920



# ==========================
# SETTINGS
# ==========================

FONT_SIZE = 62

TEXT_WIDTH = 26

POSITION_Y = 0.70


FADE_TIME = 0.35




# ==========================
# CREATE SUBTITLE
# ==========================

def create_subtitle(
        text,
        duration
):


    img = Image.new(
        "RGBA",
        (WIDTH, HEIGHT),
        (0,0,0,0)
    )


    draw = ImageDraw.Draw(
        img
    )



    font = ImageFont.truetype(
        FONT_PATH,
        FONT_SIZE
    )



    # перенос строк

    lines = textwrap.fill(
        text,
        width=TEXT_WIDTH
    )



    bbox = draw.multiline_textbbox(
        (0,0),
        lines,
        font=font,
        align="center",
        spacing=10
    )



    tw = bbox[2]-bbox[0]

    th = bbox[3]-bbox[1]



    x = (
        WIDTH - tw
    ) / 2



    y = (
        HEIGHT * POSITION_Y
    )



    # ======================
    # мягкая тень
    # ======================


    shadow = Image.new(
        "RGBA",
        (WIDTH,HEIGHT),
        (0,0,0,0)
    )


    shadow_draw = ImageDraw.Draw(
        shadow
    )


    shadow_draw.multiline_text(

        (x+4,y+4),

        lines,

        font=font,

        fill=(0,0,0,180),

        align="center",

        spacing=10

    )



    shadow = shadow.filter(
        ImageFilter.GaussianBlur(4)
    )



    img.alpha_composite(
        shadow
    )



    # ======================
    # основной текст
    # ======================


    draw.multiline_text(

        (x,y),

        lines,

        font=font,

        fill=(255,255,255,255),

        align="center",

        spacing=10

    )



    clip = ImageClip(
        np.array(img),
        duration=duration
    )



    # ======================
    # fade
    # ======================


    clip = clip.with_effects(
        [
            lambda c:
            c.with_opacity(
                lambda t:
                min(
                    1,
                    t / FADE_TIME
                )
                if t < FADE_TIME

                else

                min(
                    1,
                    (duration-t)/FADE_TIME
                )

            )
        ]
    )


    return clip
