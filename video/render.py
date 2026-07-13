import os
import requests
import asyncio
import edge_tts

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    TextClip
)


PEXELS_KEY = os.getenv("PEXELS_KEY")


async def create_voice(text):

    voice = edge_tts.Communicate(
        text,
        "ru-RU-DmitryNeural"
    )

    await voice.save(
        "output/voice.mp3"
    )



def download_video(query):

    headers = {
        "Authorization": PEXELS_KEY
    }


    url = (
        "https://api.pexels.com/videos/search"
    )


    params = {
        "query": query,
        "per_page": 1
    }


    r = requests.get(
        url,
        headers=headers,
        params=params
    )


    data = r.json()


    link = (
        data["videos"][0]
        ["video_files"][0]
        ["link"]
    )


    video = requests.get(link)


    with open(
        "output/source.mp4",
        "wb"
    ) as f:

        f.write(video.content)



def create_video(script):


    os.makedirs(
        "output",
        exist_ok=True
    )


    print("🎙 Создаём голос")


    text = script.split(
        "TEXT:"
    )[1].split(
        "SEARCH:"
    )[0]


    asyncio.run(
        create_voice(text)
    )


    print("🎥 Скачиваем видео")


    download_video(
        "science technology"
    )


    print("🎬 Монтаж")


    video = VideoFileClip(
        "output/source.mp4"
    )


    audio = AudioFileClip(
        "output/voice.mp3"
    )


    video = video.subclipped(
        0,
        min(
            video.duration,
            audio.duration
        )
    )


    video = video.resized(
        height=1920
    )


    final = video.with_audio(
        audio
    )


    final.write_videofile(
        "output/short.mp4",
        fps=30
    )


    return "output/short.mp4"
