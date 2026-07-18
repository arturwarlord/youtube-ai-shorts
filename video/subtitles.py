from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

from moviepy import ImageClip
from moviepy.video.fx import FadeIn, FadeOut



FONT_PATH = "assets/fonts/Montserrat-ExtraBold.ttf"


WIDTH = 1080
HEIGHT = 1920


FONT_SIZE = 70

POSITION_Y = 0.65

FADE_TIME = 0.15



def create_text_image(text):


    img = Image.new(

        "RGBA",

        (WIDTH, HEIGHT),

        (0,0,0,0)

    )


    draw = ImageDraw.Draw(img)



    font = ImageFont.truetype(

        FONT_PATH,

        FONT_SIZE

    )



    bbox = draw.textbbox(

        (0,0),

        text,

        font=font

    )



    w = bbox[2] - bbox[0]

    h = bbox[3] - bbox[1]



    x = (WIDTH - w) / 2


    y = HEIGHT * POSITION_Y



    # тень

    shadow = Image.new(

        "RGBA",

        (WIDTH,HEIGHT),

        (0,0,0,0)

    )


    shadow_draw = ImageDraw.Draw(
        shadow
    )


    shadow_draw.text(

        (x+5,y+5),

        text,

        font=font,

        fill=(0,0,0,180)

    )


    shadow = shadow.filter(
        ImageFilter.GaussianBlur(5)
    )


    img.alpha_composite(
        shadow
    )



    # текст

    draw.text(

        (x,y),

        text,

        font=font,

        fill=(255,255,255,255)

    )


    return np.array(img)




def create_subtitle(words):


    clips = []


    for item in words:


        text = item["word"].upper()


        start = item["start"]

        end = item["end"]


        duration = end - start



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


        clip = clip.with_effects(

            [

                FadeIn(
                    FADE_TIME
                ),

                FadeOut(
                    FADE_TIME
                )

            ]

        )


        clips.append(
            clip
        )



    return clips
