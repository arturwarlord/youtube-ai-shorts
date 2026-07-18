import os
import requests
from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv(
    "FREESOUND_API_KEY"
)


BASE_URL = "https://freesound.org/apiv2"



def search_music(query):

    if not API_KEY:
        print("❌ Нет FREESOUND_API_KEY")
        return None


    url = f"{BASE_URL}/search/text/"


    params = {

        "query": query,

        "token": API_KEY,

        "fields": "id,name,previews,tags,duration",

        "page_size": 10

    }


    response = requests.get(
        url,
        params=params,
        timeout=30
    )


    if response.status_code != 200:

        print(
            "❌ Freesound ошибка:",
            response.text
        )

        return None



    data = response.json()


    results = data.get(
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
        "🎵 Найден трек:",
        track["name"]
    )


    return {

        "name": track["name"],

        "url": track["previews"]["preview-hq-mp3"],

        "duration": track["duration"]

    }
