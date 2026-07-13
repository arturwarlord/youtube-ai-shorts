import os
import asyncio
import requests
import edge_tts

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips
)

from moviepy.video.fx import Crop

from dotenv import load_dotenv


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



# -------------------------
# VOICE
# -------------------------

async def create_voice(
    text,
    filename
):

    if isinstance(text, list):

        text = " ".join(
            scene["text"]
            for scene in text
        )


    voice = edge_tts.Communicate(
        text=text,
        voice="ru-RU-DmitryNeural"
    )


    await voice.save(
        filename
    )



# -------------------------
# PEXELS SEARCH
# -------------------------

def search_video(
    query,
    index
):

    print(
        f"🔎 Поиск: {query}"
    )


    url = (
        "https://api.pexels.com/videos/search"
    )


    headers = {

        "Authorization":
        PEXELS_KEY

    }


    params = {

        "query": query,

        "per_page": 10,

        "orientation":
        "portrait"

    }


    r = requests.get(
        url,
        headers=headers,
        params=params
    )


    data = r.json()



    if not data.get(
        "videos"
    ):

        raise Exception(
            f"Видео не найдено: {query}"
        )


    video = data["videos"][0]


    files = video["video_files"]


    best = max(
        files,
        key=lambda x:
        x.get("width",0)
    )


    link = best["link"]



    filename = (
        f"{TEMP_DIR}/scene_{index}.mp4"
    )



    video_data = requests.get(
        link
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



# -------------------------
# VIDEO PREPARE
# -------------------------

def prepare_clip(
    filename,
    duration
):


    clip = VideoFileClip(
        filename
    )


    clip = clip.resize(
        height=HEIGHT
    )


    if clip.w < WIDTH:

        clip = clip.resize(
            width=WIDTH
        )


    clip = Crop(
    width=WIDTH,
    height=HEIGHT,
    x_center=clip.w / 2,
    y_center=clip.h / 2
    ).apply(clip)


    if clip.duration < duration:

        clip = clip.loop(
            duration=duration
        )


    else:

        clip = clip.subclip(
            0,
            duration
        )


    return clip



# -------------------------
# MAIN RENDER
# -------------------------

def create_video(
    scenes
):


    voice_file = (
        f"{TEMP_DIR}/voice.mp3"
    )


    print(
        "🎙 Создание голоса"
    )


    full_text = [
        scene["text"]
        for scene in scenes
    ]



    asyncio.run(
        create_voice(
            full_text,
            voice_file
        )
    )



    audio = AudioFileClip(
        voice_file
    )


    total_time = (
        audio.duration
    )


    print(
        f"⏱ Длина голоса: {total_time:.2f} сек"
    )



    scene_time = (
        total_time /
        len(scenes)
    )



    clips = []



    print(
        "🎥 Поиск видео"
    )


    for i, scene in enumerate(
        scenes,
        start=1
    ):


        query = scene["search"]


        video_file = search_video(
            query,
            i
        )


        clip = prepare_clip(
            video_file,
            scene_time
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



    final = final.set_audio(
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
