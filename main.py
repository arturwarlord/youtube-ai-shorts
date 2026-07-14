from dotenv import load_dotenv

from ai.script import create_script
from ai.parser import parse_scenes

from video.render import create_video



load_dotenv()



def main():


    print(
        "\n🚀 Запуск AI Shorts генератора\n"
    )


    # ==========================
    # SCRIPT
    # ==========================

    script = create_script()



    # ==========================
    # PARSE
    # ==========================

    print(
        "\n🧩 Разбор сценария..."
    )


    scenes = parse_scenes(
        script
    )


    if not scenes:

        raise Exception(
            "❌ Не удалось получить сцены"
        )


    print(
        f"✅ Найдено сцен: {len(scenes)}\n"
    )



    # ==========================
    # VIDEO
    # ==========================

    print(
        "🎬 Создание видео...\n"
    )


    create_video(
        scenes
    )



    print(
        "\n🎉 Работа завершена"
    )




if __name__ == "__main__":

    main()
