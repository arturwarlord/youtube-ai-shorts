import os
import google.generativeai as genai


genai.configure(
    api_key=os.getenv("GEMINI_KEY")
)


def create_script():

    model = genai.GenerativeModel(
        "gemini-2.0-flash"
    )


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
3 слова для поиска видео на Pexels

DESCRIPTION:
описание ролика

"""


    response = model.generate_content(prompt)


    return response.text
