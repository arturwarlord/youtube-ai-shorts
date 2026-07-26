import os
import json

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload"
]


def get_youtube_service():

    token_json = os.getenv("YOUTUBE_TOKEN")

    if not token_json:

        raise Exception(
            "❌ YOUTUBE_TOKEN не найден"
        )


    credentials = Credentials.from_authorized_user_info(
        json.loads(token_json),
        SCOPES
    )


    youtube = build(
        "youtube",
        "v3",
        credentials=credentials
    )


    return youtube


def upload_video(
    video_path,
    thumbnail_path,
    title,
    description,
    hashtags,
    tags
):

    print(
        "\n📤 Загрузка видео на YouTube..."
    )


    youtube = get_youtube_service()


    # ==========================
    # DESCRIPTION
    # ==========================

    full_description = f"""
{description}

{hashtags}
""".strip()


    # ==========================
    # VIDEO METADATA
    # ==========================

    body = {

        "snippet": {

            "title": title,

            "description": full_description,

            "tags": tags,

            "categoryId": "22"

        },

        "status": {

            "privacyStatus": "public",

            "selfDeclaredMadeForKids": False

        }

    }


    # ==========================
    # UPLOAD VIDEO
    # ==========================

    media = MediaFileUpload(

        video_path,

        mimetype="video/mp4",

        resumable=True

    )


    request = youtube.videos().insert(

        part="snippet,status",

        body=body,

        media_body=media

    )


    response = None


    while response is None:

        status, response = request.next_chunk()


        if status:

            progress = int(
                status.progress() * 100
            )


            print(
                f"📤 Загрузка: {progress}%"
            )


    video_id = response["id"]


    print(
        f"\n✅ Видео загружено!"
    )


    print(
        f"🎬 Video ID: {video_id}"
    )


    print(
        f"🔗 https://www.youtube.com/watch?v={video_id}"
    )


    # ==========================
    # THUMBNAIL
    # ==========================

    if (

        thumbnail_path

        and os.path.exists(
            thumbnail_path
        )

    ):

        print(
            "\n🖼️ Загрузка обложки..."
        )


        thumbnail_media = MediaFileUpload(

            thumbnail_path,

            mimetype="image/jpeg"

        )


        youtube.thumbnails().set(

            videoId=video_id,

            media_body=thumbnail_media

        ).execute()


        print(
            "✅ Обложка установлена"
        )


    return video_id
