import os
import asyncio
import requests
import random
import edge_tts

from dotenv import load_dotenv

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip
)

from moviepy.video.fx import Crop


from video.subtitles import create_subtitle
from video.music import add_background_music

from video.freesound import search_music
from video.music_downloader import download_music

from ai.music_selector import select_music_style

from audio.whisper import transcribe_audio

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
# VOICE
# ==========================

async def create_voice(text, filename):

    if isinstance(text, list):

        text = " ".join(text)


    voice = edge_tts.Communicate(

        text=text,

        voice="ru-RU-DmitryNeural"

    )


    await voice.save(
        filename
    )





# ==========================
# SEARCH PEXELS
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

        if f.get("width", 0) >= 720

    ]


    if not files:

        files = video["video_files"]



    file = max(

        files,

        key=lambda x: x.get("width", 0)

    )


    link = file["link"]


    filename = (
        f"{TEMP_DIR}/clip_{index}.mp4"
    )


    content = requests.get(

        link,

        timeout=60

    ).content


    with open(filename, "wb") as f:

        f.write(content)


    print(
        f"⬇ Загружено видео {index}"
    )


    return filename





# ==========================
# PREPARE CLIP
# ==========================

def prepare_clip(filename, duration):


    clip = VideoFileClip(
        filename
    )


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


    # оставляем твой zoom эффект

    try:

        clip = clip.resized(

            lambda t:

            1 + (0.02 * t)

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
# SPLIT WORDS
# ==========================


def split_words_by_scenes(words, scenes):

    total_words = len(words)

    words_per_scene = total_words // len(scenes)


    result = []

    start = 0


    for index, scene in enumerate(scenes):

        if index == len(scenes) - 1:

            scene_words = words[start:]

        else:

            scene_words = words[
                start:start + words_per_scene
            ]


        result.append(
            scene_words
        )

        start += words_per_scene


    return result



# ==========================
# CREATE VIDEO
# ==========================

def create_video(
        scenes,
        music_style=None
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



    # ==========================
    # AUTO MUSIC SEARCH
    # ==========================


    full_text = " ".join(texts)


    music_file = None


    music_query = select_music_style(
        full_text
    )


    print(
        f"🎼 Определён стиль музыки: {music_query}"
    )


    track = search_music(
        music_query
    )


    if track:

        music_file = download_music(
            track
        )




    asyncio.run(

        create_voice(

            texts,

            voice_file

        )

    )



    audio = AudioFileClip(
        voice_file
    )

    words = transcribe_audio(
        voice_file
    )


    scene_words = split_words_by_scenes(
        words,
        scenes
    )


    total_duration = audio.duration


    print(
        f"⏱ Длина голоса: {total_duration:.2f} сек"
    )



    amount = len(scenes) * 2


    clip_duration = (

        total_duration / amount

    )



    clips = []


    counter = 1


    print(
        "🎥 Поиск видео"
    )



    for scene_index, scene in enumerate(scenes):

        current_words = scene_words[scene_index]

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
                current_words
            )


            clip = CompositeVideoClip(

                [
            
                    clip,
            
                    *subtitle
            
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



    print(
        "🎵 Добавление музыки"
    )


    final_audio = add_background_music(

        audio,

        total_duration,

        style=music_file if music_file else "dark",

        volume=0.12

    )


    final = final.with_audio(
        final_audio
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

        audio_bitrate="192k",

        preset="medium",

        threads=2

    )


    print(
        f"✅ Готово: {output}"
    )
