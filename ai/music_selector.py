def select_music_style(text):

    text = text.lower()


    # SCIENCE / TECHNOLOGY
    science_words = [
        "мозг",
        "нейрон",
        "учён",
        "наука",
        "исслед",
        "технолог",
        "робот",
        "искусственный интеллект",
        "космос",
        "вселен"
    ]


    # HORROR / MYSTERY
    horror_words = [
        "страш",
        "ужас",
        "тайна",
        "загад",
        "монстр",
        "пуга",
        "смерть",
        "тёмн"
    ]


    # HISTORY / DOCUMENTARY
    history_words = [
        "история",
        "древн",
        "война",
        "цивилиза",
        "король",
        "импер"
    ]


    # MOTIVATION
    motivation_words = [
        "успех",
        "деньги",
        "цель",
        "мотива",
        "достичь"
    ]



    if any(
        word in text
        for word in science_words
    ):

        return "dark ambient cinematic"



    if any(
        word in text
        for word in horror_words
    ):

        return "horror dark cinematic"



    if any(
        word in text
        for word in history_words
    ):

        return "epic documentary"



    if any(
        word in text
        for word in motivation_words
    ):

        return "inspiring cinematic"



    return "cinematic ambient"
