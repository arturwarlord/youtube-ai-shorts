import os
from google import genai


GEMINI_KEY = os.getenv(
    "GEMINI_KEY"
)


client = genai.Client(
    api_key=GEMINI_KEY
)



def create_script():


    print(
        "🤖 Создание сценария..."
    )


    prompt = """

Ты профессиональный сценарист YouTube Shorts.

Создай вирусный ролик длительностью 60 секунд.

Тематика:
интересные факты, наука, космос, технологии, история, загадки.

Структура ответа строго такая:

TITLE:
название


SCENE 1:
TEXT:
текст озвучки для этой сцены

SEARCH:
английский запрос для поиска видео Pexels


SCENE 2:
TEXT:
текст озвучки

SEARCH:
запрос


SCENE 3:
TEXT:
текст озвучки

SEARCH:
запрос


Продолжи до SCENE 8.


Важные правила:

- каждая сцена 5-8 секунд
- SEARCH только на английском
- визуал должен соответствовать тексту
- не используй общие запросы
- пиши только готовый сценарий


Пример:

SCENE 1:

TEXT:
Учёные обнаружили, что Вселенная расширяется быстрее.

SEARCH:
expanding universe galaxy stars


SCENE 2:

TEXT:
Телескопы позволяют увидеть прошлое.

SEARCH:
space telescope astronomy


"""



    models = [

        "gemini-flash-lite-latest",

        "gemini-2.5-flash",

        "gemini-flash-latest"

    ]



    for model in models:


        try:


            print(
                "Пробуем модель:",
                model
            )


            response = client.models.generate_content(

                model=model,

                contents=prompt

            )


            return response.text



        except Exception as e:


            print(
                "Ошибка модели",
                model,
                e
            )



    raise Exception(
        "Все модели Gemini недоступны"
    )
