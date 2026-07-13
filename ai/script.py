import os
from google import genai


client = genai.Client(
    api_key=os.getenv("GEMINI_KEY")
)


def create_script():

    prompt = """
Ты профессиональный сценарист YouTube Shorts.

Создай один вирусный ролик длительностью 45-60 секунд.

Тематика:
интересные факты, наука, история, загадки.

Ответ строго в формате:

TITLE:
(название ролика)

TEXT:
(текст диктора, примерно 120-150 слов)

SEARCH:
(3-5 английских слов для поиска видео на Pexels)

DESCRIPTION:
(описание для YouTube)

Правила:
- пиши на русском языке
- первые 3 секунды должны цеплять внимание
- без приветствий
- без лишней воды
"""


    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )


    return response.text
