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
# VOICE GENERATION
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
        "🔎 Поиск:",
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

        "per_page": 15,

        "orientation": "portrait"

    }


    response = requests.get(
        url,
        headers=headers,
        params=params
    )


    if response.status_code != 200:

        raise Exception(
            response.text
        )


    return response.json().get(
        "videos",
        []
    )



# ==========================
# DOWNLOAD VIDEOS
# ==========================

def download_clips(videos):

    files = []


    for index, video in enumerate(videos):


        if len(files) >= 12:

            break



        print(
            f"⬇️ Загружаем сцену {len(files)+1}/12"
        )


        candidates = video.get(
            "video_files",
            []
        )


        # только вертикальные видео

        candidates = [

            x for x in candidates

            if x.get(
                "height",
                0
            ) >= 1000

        ]



        if not candidates:

            continue



        # самое качественное

        selected = sorted(

            candidates,

            key=lambda x:

                x.get(
                    "height",
                    0
                ),

            reverse=True

        )[0]



        link = selected["link"]



        content = requests.get(
            link
        ).content



        filename = (
            f"{OUTPUT}/clip_{len(files)}.mp4"
        )



        with open(
            filename,
            "wb"
        ) as file:

            file.write(
                content
            )


        files.append(
            filename
        )



    if not files:

        raise Exception(
            "Не удалось скачать видео"
        )


    return files



# ==========================
# PREPARE SCENES
# ==========================

def prepare_scenes(files):

    scenes = []


    for file in files:


        try:

            clip = VideoFileClip(
                file
            )


            clip = clip.with_fps(
                30
            )


            clip = clip.resized(
                height=1920
            )


            duration = min(
                5,
                clip.duration
            )


            if duration > 2:


                clip = clip.subclipped(

                    0,

                    duration

                )


                scenes.append(
                    clip
                )


        except Exception as e:

            print(
                "Ошибка клипа:",
                e
            )



    return scenes



# ==========================
# MAIN VIDEO CREATION
# ==========================

def create_video(script):


    print(
        "🎙 Создание голоса"
    )



    try:

        text = (

            script

            .split(
                "TEXT:"
            )[1]

            .split(
                "SEARCH:"
            )[0]

        )


    except:


        text = script




    asyncio.run(
        create_voice(text)
    )



    audio = AudioFileClip(
        f"{OUTPUT}/voice.mp3"
    )



    print(
        "⏱ Длина голоса:",
        round(
            audio.duration,
            2
        ),
        "сек"
    )



    print(
        "🎥 Поиск видео"
    )



    videos = search_videos(
        "science space technology"
    )



    files = download_clips(
        videos
    )



    print(
        "✂️ Подготовка сцен"
    )



    scenes = prepare_scenes(
        files
    )



    if not scenes:

        raise Exception(
            "Нет подготовленных сцен"
        )



    # расширяем сцены до длины голоса

    total = sum(
        x.duration
        for x in scenes
    )



    while total < audio.duration:


        print(
            "➕ Добавляем сцены"
        )


        scenes.extend(
            scenes[:3]
        )


        total = sum(
            x.duration
            for x in scenes
        )



    print(
        "🔗 Склейка видео"
    )


    video = concatenate_videoclips(

        scenes,

        method="compose"

    )



    video = video.subclipped(

        0,

        audio.duration

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
