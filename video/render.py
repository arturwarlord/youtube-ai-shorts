import os
import requests
import asyncio
import edge_tts

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips
)


PEXELS_KEY = os.getenv("PEXELS_KEY")


OUTPUT = "output"


os.makedirs(
    OUTPUT,
    exist_ok=True
)


# ==========================
# AI VOICE
# ==========================

async def create_voice(text):

    voice = edge_tts.Communicate(
        text,
        "ru-RU-DmitryNeural"
    )

    await voice.save(
        f"{OUTPUT}/voice.mp3"
    )



# ==========================
# PEXELS SEARCH
# ==========================

def search_videos(query):

    print(
        "🔎 Ищем видео:",
        query
    )


    headers = {
        "Authorization": PEXELS_KEY
    }


    url = (
        "https://api.pexels.com/videos/search"
    )


    params = {

        "query": query,

        "per_page": 10,

        "orientation": "portrait"

    }


    response = requests.get(
        url,
        headers=headers,
        params=params
    )


    data = response.json()


    return data.get(
        "videos",
        []
    )



# ==========================
# DOWNLOAD CLIPS
# ==========================

def download_clips(videos):

    files = []


    for index, video in enumerate(videos[:6]):


        print(
            f"⬇️ Видео {index+1}/6"
        )


        candidates = (
            video["video_files"]
        )


        # выбираем лучшее подходящее видео

        file = sorted(
            candidates,
            key=lambda x: x.get(
                "width",
                0
            ),
            reverse=True
        )[0]


        link = file["link"]


        content = requests.get(
            link
        ).content



        filename = (
            f"{OUTPUT}/clip_{index}.mp4"
        )


        with open(
            filename,
            "wb"
        ) as f:

            f.write(content)


        files.append(
            filename
        )


    return files



# ==========================
# VIDEO BUILD
# ==========================

def create_video(script):


    print(
        "🎙 Создание голоса"
    )


    try:

        text = (
            script
            .split("TEXT:")[1]
            .split("SEARCH:")[0]
        )

    except:


        text = script



    asyncio.run(
        create_voice(text)
    )



    print(
        "🎥 Поиск сцен"
    )


    videos = search_videos(
        "space science technology"
    )


    if not videos:

        raise Exception(
            "Pexels не нашел видео"
        )



    files = download_clips(
        videos
    )



    print(
        "✂️ Подготовка сцен"
    )


    scenes = []



    for file in files:


        clip = VideoFileClip(
            file
        )


        # фиксируем FPS

        clip = clip.with_fps(
            30
        )


        # вертикальный формат

        clip = clip.resized(
            height=1920
        )


        # обрезаем лишнее

        duration = min(
            6,
            clip.duration
        )


        clip = clip.subclipped(
            0,
            duration
        )


        scenes.append(
            clip
        )



    print(
        "🔗 Склейка сцен"
    )


    video = concatenate_videoclips(
        scenes,
        method="compose"
    )



    print(
        "🎧 Добавляем голос"
    )


    audio = AudioFileClip(
        f"{OUTPUT}/voice.mp3"
    )



    duration = min(
        video.duration,
        audio.duration
    )


    video = video.subclipped(
        0,
        duration
    )


    video = video.with_audio(
        audio
    )



    print(
        "💾 Рендер"
    )


    video.write_videofile(

        f"{OUTPUT}/short.mp4",

        fps=30,

        codec="libx264",

        audio_codec="aac",

        bitrate="5000k",

        preset="medium",

        threads=2

    )



    print(
        "✅ Готово:",
        f"{OUTPUT}/short.mp4"
    )


    return (
        f"{OUTPUT}/short.mp4"
    )
