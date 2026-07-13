import os
from google import genai


client = genai.Client(
    api_key=os.getenv("GEMINI_KEY")
)


def create_script():

    prompt = """
Создай сценарий YouTube Shorts.

Тема:
интересные факты.

Формат:

TITLE:
название

TEXT:
текст диктора на 45 секунд

SEARCH:
ключевые слова для поиска видео

DESCRIPTION:
описание

Пиши на русском языке.
"""


    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )


    return response.text
