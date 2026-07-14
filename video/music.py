import os

from moviepy import (
    AudioFileClip,
    CompositeAudioClip
)



MUSIC_DIR = "assets/music"



def load_music(style="dark"):


    music_file = (
        f"{MUSIC_DIR}/{style}.mp3"
    )


    if not os.path.exists(music_file):

        music_file = (
            f"{MUSIC_DIR}/background.mp3"
        )



    if not os.path.exists(music_file):

        print(
            "⚠ Музыка не найдена"
        )

        return None



    print(
        f"🎵 Загружена музыка: {music_file}"
    )



    return AudioFileClip(
        music_file
    )





def add_background_music(
        voice,
        duration,
        style="dark",
        volume=0.12
):


    music = load_music(
        style
    )



    if music is None:

        return voice




    if music.duration < duration:


        music = music.loop(
            duration=duration
        )


    else:


        music = music.subclipped(
            0,
            duration
        )




    music = music.with_volume_scaled(
        volume
    )




    return CompositeAudioClip(
        [
            voice,
            music
        ]
    )
