import os
from google import genai


client = genai.Client(
    api_key=os.getenv("GEMINI_KEY")
)


def create_script():

    prompt = """
Ты профессиональный сценарист вирусных YouTube Shorts.

Создай сценарий ролика 55-60 секунд.

Темы:
- космос
- загадки науки
- история
- технологии
- невероятные факты


Структура:

TITLE:
Короткий кликабельный заголовок.


HOOK:
Первые 3 секунды.
Должно заставить человека остановить скролл.


TEXT:
Полный текст диктора.
120-160 слов.
Без таймкодов.


SEARCH:
5 английских поисковых запросов для Pexels.


KEYWORDS:
5 главных слов для выделения в субтитрах.


DESCRIPTION:
Описание YouTube Shorts.


Правила:
- русский язык
- без приветствия
- никаких "сегодня мы узнаем"
- стиль Netflix documentary
- интрига в начале
"""


    models = [
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite"
    ]


    for model in models:

        try:

            print("Пробуем модель:", model)

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
        "Нет доступных Gemini моделей"
    )
