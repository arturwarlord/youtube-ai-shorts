import os

from moviepy import (
    AudioFileClip,
    CompositeAudioClip
)

from moviepy.audio.fx import AudioLoop


MUSIC_DIR = "assets/music"



def load_music(style="dark"):


    # Если передали уже готовый путь к файлу
    # например assets/music/cache/music.mp3

    if os.path.exists(style):

        print(
            f"🎵 Загружена музыка: {style}"
        )

        return AudioFileClip(
            style
        )



    # Старый режим:
    # dark -> assets/music/dark.mp3

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


    music = AudioLoop(
        duration=duration
    ).apply(
        music
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
