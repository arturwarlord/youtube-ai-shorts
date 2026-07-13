import os
from google import genai


client = genai.Client(
    api_key=os.getenv("GEMINI_KEY")
)


def create_script():

    print("Проверяем доступные модели...")

    models = client.models.list()

    for model in models:
        print(model.name)


    return "test"
