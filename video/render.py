import os
import asyncio
import requests
import random
import edge_tts

from dotenv import load_dotenv

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips
)

from moviepy import CompositeVideoClip
from video.subtitles import create_subtitle

from moviepy.video.fx import Crop


load_dotenv()


PEXELS_KEY = os.getenv("PEXELS_KEY")


OUTPUT_DIR = "output"
TEMP_DIR = "temp"


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


WIDTH = 1080
HEIGHT = 1920



# ==========================
# VOICE
# ==========================

async def create_voice(text, filename):

    if isinstance(text, list):
        text = " ".join(text)

    voice = edge_tts.Communicate(
        text=text,
        voice="ru-RU-DmitryNeural"
    )

    await voice.save(filename)



# ==========================
# SEARCH PEXELS
# ==========================

def search_video(query, index):

    print(f"🔎 Поиск: {query}")


    url = "https://api.pexels.com/videos/search"


    headers = {
        "Authorization": PEXELS_KEY
    }


    params = {

        "query": query,

        "per_page": 15,

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


    videos = data["videos"]


    video = random.choice(
        videos[:10]
    )


    files = video["video_files"]


    files = [
        f for f in files
        if f.get("width",0) >= 720
    ]


    if not files:
        files = video["video_files"]


    file = max(
        files,
        key=lambda x:x.get("width",0)
    )


    link = file["link"]



    filename = (
        f"{TEMP_DIR}/clip_{index}.mp4"
    )


    data = requests.get(
        link,
        timeout=60
    ).content


    with open(filename,"wb") as f:
        f.write(data)


    print(
        f"⬇️ Загружено видео {index}"
    )


    return filename



# ==========================
# PREPARE CLIP
# ==========================

def prepare_clip(
    filename,
    duration
):


    clip = VideoFileClip(filename)



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
        x_center=clip.w/2,
        y_center=clip.h/2
    ).apply(
        clip
    )



    # немного движения камеры

    try:

        clip = clip.resized(
            lambda t:
            1 + (0.02*t)
        )

    except:

        pass



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
# CREATE VIDEO
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
        s["text"]
        for s in scenes
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


    total_duration = audio.duration


    print(
        f"⏱ Длина голоса: {total_duration:.2f} сек"
    )



    # делаем больше кадров

    amount = len(scenes) * 2


    clip_duration = (
        total_duration / amount
    )



    clips = []
    subtitle_clips = []


    print(
        "🎥 Поиск видео"
    )



    counter=1


    for scene in scenes:


        for part in range(2):


            video_file = search_video(
                scene["search"],
                counter
            )


            clip = prepare_clip(
                video_file,
                clip_duration
            )

            subtitle = create_subtitle(
                scene["text"],
                clip_duration
            )
            
            clip = CompositeVideoClip(
                [
                    clip,
                    subtitle
                ]
            )


            clips.append(
                clip
            )


            counter += 1



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
