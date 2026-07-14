from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap

from moviepy import ImageClip


FONT_PATH = "assets/fonts/Montserrat-ExtraBold.ttf"

WIDTH = 1080
HEIGHT = 1920


def create_subtitle(text, duration):

    img = Image.new(
        "RGBA",
        (WIDTH, HEIGHT),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(
        FONT_PATH,
        78
    )

    # автоматический перенос строк
    lines = textwrap.fill(
        text.upper(),
        width=18
    )

    bbox = draw.multiline_textbbox(
        (0, 0),
        lines,
        font=font,
        align="center"
    )

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (WIDTH - tw) / 2
    y = HEIGHT * 0.68

    # черная обводка
    stroke = 5

    draw.multiline_text(
        (x, y),
        lines,
        font=font,
        fill="white",
        align="center",
        stroke_width=stroke,
        stroke_fill="black"
    )

    return ImageClip(
        np.array(img),
        duration=duration
    )
