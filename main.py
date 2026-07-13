from ai.script import create_script
from ai.parser import parse_scenes

from video.render import create_video



def main():


    print(
        "🤖 Создание сценария..."
    )


    script = create_script()



    print(
        "\n📄 Получен сценарий:\n"
    )


    print(
        script
    )



    print(
        "\n🧩 Разбор сцен..."
    )


    scenes = parse_scenes(
        script
    )



    print(
        f"Найдено сцен: {len(scenes)}"
    )



    for i, scene in enumerate(
        scenes,
        start=1
    ):

        print(
            f"\nСцена {i}"
        )

        print(
            "Текст:",
            scene["text"]
        )

        print(
            "Поиск:",
            scene["search"]
        )



    print(
        "\n🎬 Создание видео..."
    )


    create_video(
        scenes
    )



if __name__ == "__main__":

    main()
