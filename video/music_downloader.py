import os
import requests


CACHE_DIR = "assets/music/cache"


os.makedirs(
    CACHE_DIR,
    exist_ok=True
)



def download_music(track):

    filename = (
        track["name"]
        .replace(" ","_")
        .replace("/","")
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
        "⬇ Скачивание музыки"
    )


    response = requests.get(
        track["url"],
        timeout=60
    )


    with open(
        path,
        "wb"
    ) as file:

        file.write(
            response.content
        )


    return path
