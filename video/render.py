import os
import asyncio
import requests
import edge_tts

from dotenv import load_dotenv

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips
)

from moviepy.video.fx import Crop


load_dotenv()


PEXELS_KEY = os.getenv("PEXELS_KEY")


OUTPUT_DIR = "output"
TEMP_DIR = "temp"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    TEMP_DIR,
    exist_ok=True
)


WIDTH = 1080
HEIGHT = 1920



# ==========================
# CREATE VOICE
# ==========================

async def create_voice(text, filename):

    if isinstance(text, list):

        text = " ".join(
            text
        )


    voice = edge_tts.Communicate(
        text=text,
        voice="ru-RU-DmitryNeural"
    )


    await voice.save(
        filename
    )



# ==========================
# PEXELS SEARCH
# ==========================

def search_video(query, index):

    print(
        f"🔎 Поиск: {query}"
    )


    url = (
        "https://api.pexels.com/videos/search"
    )


    headers = {
        "Authorization": PEXELS_KEY
    }


    params = {

        "query": query,

        "per_page": 10,

        "orientation": "portrait"

    }


    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )


    data = response.json()


    if not data.get("videos"):

        raise Exception(
            f"Видео не найдено: {query}"
        )


    video = data["videos"][0]


    files = video["video_files"]


    file = max(
        files,
        key=lambda x:
        x.get("width", 0)
    )


    link = file["link"]



    filename = (
        f"{TEMP_DIR}/scene_{index}.mp4"
    )


    video_data = requests.get(
        link,
        timeout=60
    ).content



    with open(
        filename,
        "wb"
    ) as f:

        f.write(
            video_data
        )


    print(
        f"⬇️ Загружена сцена {index}"
    )


    return filename



# ==========================
# PREPARE VIDEO
# ==========================

def prepare_clip(
    filename,
    duration
):

    clip = VideoFileClip(
        filename
    )


    # вертикальный формат

    clip = clip.resized(
        height=HEIGHT
    )


    if clip.w < WIDTH:

        clip = clip.resized(
            width=WIDTH
        )


    clip = Crop(
        width=WIDTH,
        height=HEIGHT,
        x_center=clip.w / 2,
        y_center=clip.h / 2
    ).apply(
        clip
    )



    if clip.duration < duration:

        clip = clip.with_duration(
            duration
        )


    else:

        clip = clip.subclipped(
            0,
            duration
        )


    return clip



# ==========================
# MAIN VIDEO
# ==========================

def create_video(
    scenes
):


    print(
        "🎙 Создание голоса"
    )


    voice_file = (
        f"{TEMP_DIR}/voice.mp3"
    )


    texts = [
        scene["text"]
        for scene in scenes
    ]


    asyncio.run(
        create_voice(
            texts,
            voice_file
        )
    )



    audio = AudioFileClip(
        voice_file
    )


    total_duration = (
        audio.duration
    )


    print(
        f"⏱ Длина голоса: {total_duration:.2f} сек"
    )



    scene_duration = (
        total_duration /
        len(scenes)
    )



    clips = []


    print(
        "🎥 Поиск видео"
    )



    for index, scene in enumerate(
        scenes,
        start=1
    ):


        video_file = search_video(
            scene["search"],
            index
        )


        clip = prepare_clip(
            video_file,
            scene_duration
        )


        clips.append(
            clip
        )



    print(
        "✂️ Склейка сцен"
    )


    final = concatenate_videoclips(
        clips,
        method="compose"
    )



    final = final.with_audio(
        audio
    )



    output = (
        f"{OUTPUT_DIR}/short.mp4"
    )


    print(
        "💾 Рендер"
    )


    final.write_videofile(

        output,

        fps=30,

        codec="libx264",

        audio_codec="aac",

        preset="medium",

        threads=2

    )


    print(
        f"✅ Готово: {output}"
    )
