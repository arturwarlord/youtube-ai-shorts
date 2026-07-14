import os

from dotenv import load_dotenv

from ai.generator import generate_script
from ai.parser import parse_scenes

from video.render import create_video



load_dotenv()



def main():

    print(
        "\n🤖 Генерация сценария..."
    )


    script = generate_script()


    scenes = parse_scenes(
        script
    )


    if not scenes:

        raise Exception(
            "❌ Сценарий пустой"
        )


    print(
        f"✅ Получено сцен: {len(scenes)}\n"
    )


    create_video(
        scenes
    )



if __name__ == "__main__":

    main()
