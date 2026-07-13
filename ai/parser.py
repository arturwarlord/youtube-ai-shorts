import re


def parse_scenes(script):

    scenes = []


    blocks = re.split(
        r"SCENE \d+:",
        script
    )


    for block in blocks:


        if "TEXT:" not in block:

            continue


        try:

            text = (
                block
                .split("TEXT:")[1]
                .split("SEARCH:")[0]
                .strip()
            )


            search = (
                block
                .split("SEARCH:")[1]
                .strip()
                .split("\n")[0]
            )


            scenes.append({

                "text": text,

                "search": search

            })


        except:

            continue



    return scenes
