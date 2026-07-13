from moviepy import TextClip


def create_caption(text):

    caption = TextClip(
        text,
        font_size=80,
        color="white",
        stroke_color="black",
        stroke_width=3,
        method="caption",
        size=(900,None)
    )

    return caption
