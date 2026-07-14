import os

from moviepy import (
    AudioFileClip,
    CompositeAudioClip
)


MUSIC_DIR = "assets/music"



# ==========================
# LOAD BACKGROUND MUSIC
# ==========================

def load_music(
        style="dark"
):


    music_file = (
        f"{MUSIC_DIR}/{style}.mp3"
    )


    if not os.path.exists(music_file):

        music_file = (
            f"{MUSIC_DIR}/background.mp3"
        )


    if not os.path.exists(music_file):

        print(
            "⚠ Музыкальный файл не найден"
        )

        return None



    print(
        f"🎵 Загружена музыка: {music_file}"
    )



    return AudioFileClip(
        music_file
    )





# ==========================
# MIX MUSIC + VOICE
# ==========================

def add_background_music(
        video,
        style="dark",
        volume=0.12
):


    music = load_music(
        style
    )



    if music is None:

        return video.audio




    # подгоняем длительность

    if music.duration < video.duration:


        music = music.loop(
            duration=video.duration
        )


    else:


        music = music.subclipped(
            0,
            video.duration
        )




    # громкость

    music = music.with_volume_scaled(
        volume
    )




    return CompositeAudioClip(
        [
            video.audio,
            music
        ]
    )
