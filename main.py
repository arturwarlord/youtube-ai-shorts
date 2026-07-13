from ai.script import create_script
from video.render import create_video


def main():

    print("🤖 Создание сценария...")

    script = create_script()

    print(script)


    print("🎬 Создание видео...")

    video = create_video(script)

    print("Готово:", video)



if __name__ == "__main__":
    main()
