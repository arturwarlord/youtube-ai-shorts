import os
import re

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

Создай вирусный ролик длительностью 60 секунд.

Тема:
случайный научный факт, загадка или тайна.

Правила:

1. Сделай 10 сцен.

2. Каждая сцена:
- 1-2 предложения текста
- длительность 5-7 секунд

3. Текст должен удерживать внимание:
- первая сцена должна иметь сильный хук
- каждые 5 секунд новая мысль
- финал должен вызвать комментарии


4. SEARCH очень важен.

SEARCH должен содержать только реальные запросы для Pexels.

Запрещено:

- alien hologram
- futuristic portal
- abstract concept
- magic effect
- impossible objects


Используй реальные объекты:

хорошо:

space galaxy stars telescope
human brain scan
storm clouds lightning
ancient ruins
ocean waves
animals
technology
laboratory


Плохие примеры:

question mark hologram
alien eye floating
cosmic energy barrier


Формат строго:


TITLE:
название


SCENE 1:

TEXT:
текст

SEARCH:
реальный запрос для видео


SCENE 2:

TEXT:
текст

SEARCH:
реальный запрос для видео


...


SCENE 10:

TEXT:
текст

SEARCH:
реальный запрос для видео


Не добавляй ничего кроме формата.

"""


    response = None


    for model in MODELS:

        try:

            print(
                f"Пробуем модель: {model}"
            )


            response = client.models.generate_content(

                model=model,

                contents=prompt

            )


            break


        except Exception as e:

            print(
                f"Ошибка модели {model}: {e}"
            )



    if response is None:

        raise Exception(
            "Все модели недоступны"
        )



    text = response.text



    print(
        "\n📄 Получен сценарий:\n"
    )

    print(text)



    return parse_script(
        text
    )





def parse_script(text):


    scenes=[]


    blocks = re.split(
        r"SCENE\s+\d+:",
        text
    )



    for block in blocks[1:]:


        text_match = re.search(

            r"TEXT:\s*(.*?)\s*SEARCH:",

            block,

            re.S

        )


        search_match = re.search(

            r"SEARCH:\s*(.*)",

            block,

            re.S

        )



        if text_match and search_match:


            scene = {


                "text":
                text_match.group(1)
                .strip(),


                "search":
                search_match.group(1)
                .strip()

            }


            scenes.append(
                scene
            )



    print(
        "\n🧩 Разбор сцен..."
    )


    print(
        f"Найдено сцен: {len(scenes)}"
    )



    for i,s in enumerate(
        scenes,
        1
    ):

        print(
            f"\nСцена {i}"
        )

        print(
            "Текст:",
            s["text"]
        )

        print(
            "Поиск:",
            s["search"]
        )



    return scenes
