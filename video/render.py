import os
import requests
import edge_tts


PEXELS_KEY = os.getenv("PEXELS_KEY")


async def voice(text):

    communicate = edge_tts.Communicate(
        text,
        "ru-RU-DmitryNeural"
    )

    await communicate.save(
        "voice.mp3"
    )



def create_video(script):

    print("Ищем видео...")


    headers = {
        "Authorization": PEXELS_KEY
    }


    url = "https://api.pexels.com/videos/search"


    params = {
        "query": "nature",
        "per_page": 1
    }


    r = requests.get(
        url,
        headers=headers,
        params=params
    )


    data = r.json()


    video_url = (
        data["videos"][0]
        ["video_files"][0]
        ["link"]
    )


    video = requests.get(video_url)


    os.makedirs(
        "output",
        exist_ok=True
    )


    with open(
        "output/background.mp4",
        "wb"
    ) as f:
        f.write(video.content)


    print("Видео скачано")

    return "output/background.mp4"
