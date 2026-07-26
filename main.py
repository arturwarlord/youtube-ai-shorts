from dotenv import load_dotenv

from ai.script import create_script
from ai.parser import parse_scenes
from ai.metadata import generate_metadata

from video.render import create_video
from video.thumbnail import create_thumbnail
from youtube.upload import upload_video

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
    # METADATA
    # ==========================

    metadata = generate_metadata(
        script
    )

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

    # ==========================
    # THUMBNAIL
    # ==========================

    thumbnail_path = create_thumbnail(
        video_path="output/short.mp4",
        title=metadata["title"]
    )

    # ==========================
    # YOUTUBE UPLOAD
    # ==========================

    print(
        "\n📤 Публикация на YouTube..."
    )


    upload_video(

        video_path="output/short.mp4",

        thumbnail_path=thumbnail_path,

        title=metadata["title"],

        description=metadata["description"],

        hashtags=metadata["hashtags"],

        tags=metadata["tags"]

    )


    print(
        "\n🎉 Работа завершена"
    )




if __name__ == "__main__":

    main()
