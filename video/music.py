import json
import random
import os


MUSIC_DIR = "assets/music"


def load_music():

    with open(
        f"{MUSIC_DIR}/music.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



def choose_music(mood=None):

    tracks = load_music()


    if mood:

        filtered = [
            x for x in tracks
            if x["mood"] == mood
        ]

        if filtered:
            return random.choice(filtered)


    return random.choice(tracks)



def get_music_path(track):

    return os.path.join(
        MUSIC_DIR,
        track["file"]
    )
