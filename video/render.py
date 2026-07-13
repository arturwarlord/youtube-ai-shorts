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


os.makedirs(
    "output",
    exist_ok=True
)



async def create_voice(text):

    communicate = edge_tts.Communicate(
        text,
        "ru-RU-DmitryNeural"
    )

    await communicate.save(
        "output/voice.mp3"
    )



def search_videos(query):

    headers = {
        "Authorization": PEXELS_KEY
    }


    url = (
        "https://api.pexels.com/videos/search"
    )


    params = {
        "query": query,
        "per_page": 5
    }


    r = requests.get(
        url,
        headers=headers,
        params=params
    )


    return r.json()["videos"]



def download_clips(videos):

    files = []


    for i, video in enumerate(videos):

        link = (
            video["video_files"][0]["link"]
        )


        data = requests.get(link).content


        filename = (
            f"output/clip_{i}.mp4"
        )


        with open(
            filename,
            "wb"
        ) as f:

            f.write(data)


        files.append(filename)


    return files




def create_video(script):


    print("🎙 Создаём голос")


    text = (
        script
        .split("TEXT:")[1]
        .split("SEARCH:")[0]
    )


    asyncio.run(
        create_voice(text)
    )



    print("🎥 Ищем несколько видео")


    clips = search_videos(
        "space science technology"
    )


    files = download_clips(
        clips
    )


    print("✂️ Собираем сцены")


    scenes = []


    for file in files:

        clip = VideoFileClip(file)


        clip = clip.resized(
            height=1920
        )


        clip = clip.subclipped(
            0,
            min(
                8,
                clip.duration
            )
        )


        scenes.append(
            clip
        )


    video = concatenate_videoclips(
        scenes,
        method="compose"
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


    video = video.with_audio(
        audio
    )


    video.write_videofile(
        "output/short.mp4",
        fps=30,
        codec="libx264",
        audio_codec="aac"
    )


    return "output/short.mp4"
