from google_auth_oauthlib.flow import InstalledAppFlow


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload"
]


flow = InstalledAppFlow.from_client_secrets_file(
    "client_secret.json",
    SCOPES
)


credentials = flow.run_local_server(
    port=8080,
    open_browser=False
)


with open("token.json", "w") as token:
    token.write(credentials.to_json())


print("✅ token.json успешно создан")
