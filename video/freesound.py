import os
import requests

from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv(
    "FREESOUND_API_KEY"
)


BASE_URL = "https://freesound.org/apiv2"



def search_music(query):

    print(
        f"🎵 Поиск музыки: {query}"
    )


    url = f"{BASE_URL}/search/text/"


    params = {

        "query": query,

        "token": API_KEY,

        "fields":
        "id,name,previews,duration,tags",

        "page_size": 15

    }


    response = requests.get(
        url,
        params=params,
        timeout=30
    )


    if response.status_code != 200:

        print(
            "❌ Freesound:",
            response.text
        )

        return None



    results = response.json().get(
        "results",
        []
    )


    if not results:

        print(
            "⚠ Музыка не найдена"
        )

        return None



    track = results[0]


    print(
        "🎧 Выбран:",
        track["name"]
    )


    return {

        "name":
        track["name"],


        "url":
        track["previews"]["preview-hq-mp3"]

    }
