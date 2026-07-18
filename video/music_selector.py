def select_music_style(text):

    text = text.lower()


    if any(word in text for word in [
        "мозг",
        "наука",
        "учёные",
        "космос",
        "технологии"
    ]):
        return "dark ambient cinematic"


    if any(word in text for word in [
        "страш",
        "ужас",
        "тайна",
        "монстр"
    ]):
        return "horror dark"


    if any(word in text for word in [
        "история",
        "древн",
        "война"
    ]):
        return "epic documentary"


    return "cinematic ambient"
