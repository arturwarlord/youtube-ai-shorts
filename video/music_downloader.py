import os
import re
import requests


CACHE_DIR = "assets/music/cache"


os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


def download_music(track):

    filename = re.sub(
        r"[^a-zA-Z0-9_-]",
        "",
        track["name"]
    )

    filename = (
        filename[:80]
        +
        ".mp3"
    )


    path = os.path.join(
        CACHE_DIR,
        filename
    )


    if os.path.exists(path):

        print(
            "🎵 Музыка из кеша"
        )

        return path



    print(
        "⬇ Скачивание музыки:",
        track["name"]
    )


    response = requests.get(
        track["url"],
        timeout=60
    )


    response.raise_for_status()


    with open(
        path,
        "wb"
    ) as file:

        file.write(
            response.content
        )


    print(
        "✅ Музыка сохранена:",
        path
    )


    return path
