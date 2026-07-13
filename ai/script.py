import os
from google import genai


client = genai.Client(
    api_key=os.getenv("GEMINI_KEY")
)


def create_script():

    prompt = """
Создай сценарий YouTube Shorts.

Тематика:
интересные факты, история, наука, загадки.

Формат:

TITLE:
Название ролика

TEXT:
Текст диктора 45-60 секунд.

SEARCH:
Ключевые слова на английском для поиска видео.

DESCRIPTION:
Описание ролика.

Требования:
- русский язык
- сильное начало первые 3 секунды
- без приветствия
- стиль как у вирусных Shorts
"""


    models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-flash-lite-latest"
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
