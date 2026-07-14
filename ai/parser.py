import re


def parse_scenes(script):

    if not isinstance(script, str):
        return script


    scenes = []


    blocks = re.split(
        r"SCENE\s+\d+:",
        script
    )


    for block in blocks:

        if not block.strip():
            continue


        text_match = re.search(
            r"TEXT:\s*(.*?)\s*SEARCH:",
            block,
            re.S
        )


        search_match = re.search(
            r"SEARCH:\s*(.*)",
            block,
            re.S
        )


        if text_match and search_match:

            text = (
                text_match
                .group(1)
                .strip()
                .replace("\n", " ")
            )


            search = (
                search_match
                .group(1)
                .strip()
                .replace("\n", " ")
            )


            scenes.append(
                {
                    "text": text,
                    "search": search
                }
            )


    return scenes
