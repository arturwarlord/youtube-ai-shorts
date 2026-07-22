import os
import json

from google_auth_oauthlib.flow import InstalledAppFlow


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload"
]


def main():

    credentials_file = "credentials.json"

    flow = InstalledAppFlow.from_client_secrets_file(
        credentials_file,
        SCOPES
    )

    credentials = flow.run_local_server(
        port=0
    )

    with open("token.json", "w") as token:

        token.write(
            credentials.to_json()
        )

    print("✅ token.json успешно создан")


if __name__ == "__main__":

    main()
