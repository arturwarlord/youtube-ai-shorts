import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_KEY")
)



MODELS = [

    "gemini-flash-lite-latest",

    "gemini-2.0-flash",

    "gemini-2.0-flash-lite"

]



def create_script():


    print(
        "🤖 Создание сценария..."
    )


    prompt = """

Ты профессиональный сценарист YouTube Shorts.

Создай научный ролик длительностью около 60 секунд.


Правила:

- ровно 10 сцен
- каждая сцена 5-7 секунд
- первая сцена должна быть мощным хуком
- последняя сцена должна вызывать комментарии


SEARCH должен быть только для Pexels.


Используй реальные запросы:

Хорошо:

space galaxy stars telescope
scientist laboratory computer
brain neurons animation
ocean waves storm
animals nature documentary


Плохо:

alien hologram
magic portal
abstract energy
future technology concept


Формат строго:


TITLE:

название


SCENE 1:

TEXT:
текст

SEARCH:
запрос


SCENE 2:

TEXT:
текст

SEARCH:
запрос


...


SCENE 10:

TEXT:
текст

SEARCH:
запрос


Не добавляй пояснений.
"""


    for model in MODELS:


        try:

            print(
                f"Пробуем модель: {model}"
            )


            response = client.models.generate_content(

                model=model,

                contents=prompt

            )


            text = response.text


            print(
                "\n📄 Получен сценарий:\n"
            )

            print(
                text
            )


            return text



        except Exception as e:

            print(
                f"Ошибка модели {model}: {e}"
            )



    raise Exception(
        "Нет доступных моделей"
    )
