import os
import json
import random

from dotenv import load_dotenv
from google import genai


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(

    api_key=os.getenv(

        "GEMINI_KEY"

    )

)


# =========================================================
# MODELS
# =========================================================

MODELS = [

    "gemini-flash-lite-latest",

    "gemini-2.0-flash",

    "gemini-2.0-flash-lite"

]


# =========================================================
# PATHS
# =========================================================

HISTORY_FILE = (

    "ai/topic_history.json"

)


# =========================================================
# SCRIPT FORMATS
# =========================================================

SCRIPT_FORMATS = [

    "SCIENTIFIC_SHOCK",

    "THOUGHT_EXPERIMENT",

    "REAL_EXPERIMENT",

    "MYTH_BUSTING",

    "HIDDEN_MECHANISM",

    "HISTORICAL_DISCOVERY",

    "FUTURE_SCENARIO",

    "DARK_QUESTION"

]


# =========================================================
# TOPIC CATEGORIES
# =========================================================

TOPIC_CATEGORIES = [

    "мозг и сознание",

    "космос и Вселенная",

    "физика повседневных явлений",

    "человеческое тело",

    "биология и эволюция",

    "животные и природа",

    "время и пространство",

    "технологии и искусственный интеллект",

    "научные эксперименты",

    "необычные явления на Земле",

    "медицина и открытия",

    "океан и глубоководный мир",

    "геология и вулканы",

    "химия в повседневной жизни",

    "генетика"

]


# =========================================================
# EMOTIONS
# =========================================================

EMOTIONS = [

    "удивление",

    "любопытство",

    "тревога",

    "восхищение",

    "шок",

    "научное любопытство",

    "ощущение тайны"

]


# =========================================================
# LOAD HISTORY
# =========================================================

def load_history():

    if not os.path.exists(

        HISTORY_FILE

    ):

        return []


    try:

        with open(

            HISTORY_FILE,

            "r",

            encoding="utf-8"

        ) as file:

            history = json.load(

                file

            )


            if isinstance(

                history,

                list

            ):

                return history


    except Exception as e:

        print(

            f"⚠️ Ошибка чтения истории: {e}"

        )


    return []


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(

        title,

        category,

        script_format,

        emotion,

        topic

):

    history = load_history()


    new_item = {

        "title": title,

        "category": category,

        "format": script_format,

        "emotion": emotion,

        "topic": topic

    }


    history.append(

        new_item

    )


    # Храним только последние 20 видео

    history = history[-20:]


    try:

        with open(

            HISTORY_FILE,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                history,

                file,

                ensure_ascii=False,

                indent=2

            )


        print(

            "💾 История темы сохранена"

        )


    except Exception as e:

        print(

            f"⚠️ Не удалось сохранить историю: {e}"

        )


# =========================================================
# FORMAT HISTORY FOR GEMINI
# =========================================================

def format_history_for_prompt(

        history

):

    if not history:

        return (

            "История отсутствует. "

            "Можно выбрать любую тему."

        )


    lines = []


    for index, item in enumerate(

        history,

        1

    ):

        title = item.get(

            "title",

            "Без названия"

        )


        category = item.get(

            "category",

            "Неизвестная категория"

        )


        script_format = item.get(

            "format",

            "Неизвестный формат"

        )


        topic = item.get(

            "topic",

            "Неизвестная тема"

        )


        lines.append(

            f"{index}. "

            f"Тема: {topic} | "

            f"Категория: {category} | "

            f"Формат: {script_format} | "

            f"Название: {title}"

        )


    return "\n".join(

        lines

    )


# =========================================================
# EXTRACT TITLE
# =========================================================

def extract_title(

        text

):

    lines = text.splitlines()


    for index, line in enumerate(

        lines

    ):

        line = line.strip()


        if line.upper() == "TITLE:":

            for next_line in lines[index + 1:]:

                next_line = next_line.strip()


                if next_line:

                    return next_line


    return "Без названия"


# =========================================================
# EXTRACT TOPIC
# =========================================================

def extract_topic(

        text

):

    lines = text.splitlines()


    for index, line in enumerate(

        lines

    ):

        line = line.strip()


        if line.upper() == "TOPIC:":

            for next_line in lines[index + 1:]:

                next_line = next_line.strip()


                if next_line:

                    return next_line


    return "Неизвестная тема"


# =========================================================
# CREATE SCRIPT
# =========================================================

def create_script():


    print(

        "🤖 Создание сценария..."

    )


    history = load_history()


    history_text = format_history_for_prompt(

        history

    )


    # -----------------------------------------------------
    # Выбираем формат, которого не было недавно
    # -----------------------------------------------------

    recent_formats = [

        item.get(

            "format"

        )

        for item in history[-3:]

    ]


    available_formats = [

        script_format

        for script_format in SCRIPT_FORMATS

        if script_format not in recent_formats

    ]


    if not available_formats:

        available_formats = SCRIPT_FORMATS


    script_format = random.choice(

        available_formats

    )


    # -----------------------------------------------------
    # Выбираем категорию, которой не было недавно
    # -----------------------------------------------------

    recent_categories = [

        item.get(

            "category"

        )

        for item in history[-4:]

    ]


    available_categories = [

        category

        for category in TOPIC_CATEGORIES

        if category not in recent_categories

    ]


    if not available_categories:

        available_categories = TOPIC_CATEGORIES


    topic_category = random.choice(

        available_categories

    )


    emotion = random.choice(

        EMOTIONS

    )


    print(

        f"🎭 Формат: {script_format}"

    )


    print(

        f"🌍 Категория: {topic_category}"

    )


    print(

        f"🎯 Эмоция: {emotion}"

    )


    # =====================================================
    # PROMPT
    # =====================================================

    prompt = f"""

Ты профессиональный сценарист научно-популярных
YouTube Shorts.

Создай один уникальный сценарий для вертикального
видео длительностью около 60 секунд.

==================================================
ТЕКУЩИЕ ПАРАМЕТРЫ
==================================================

КАТЕГОРИЯ:

{topic_category}


ФОРМАТ:

{script_format}


ЭМОЦИЯ:

{emotion}


==================================================
ИСТОРИЯ ПРЕДЫДУЩИХ ВИДЕО
==================================================

{history_text}


==================================================
ГЛАВНАЯ ЗАДАЧА
==================================================

Выбери НОВУЮ конкретную научную тему внутри
указанной категории.

Тема не должна повторять:

- предыдущие темы;
- предыдущие научные открытия;
- предыдущие эксперименты;
- предыдущие события;
- предыдущие основные идеи.

Нельзя просто взять старую тему и рассказать её
другими словами.

Если в истории уже было видео про мозг, память,
восприятие или сознание, не создавай новое видео
на ту же основную идею.

==================================================
РАЗНООБРАЗИЕ
==================================================

Не используй постоянно структуру:

"Ты думаешь X,
но на самом деле Y,
твой мозг скрывает правду,
а теперь задумайся".

ЗАПРЕЩЕНО:

- начинать каждый ролик со слова "Ты";
- постоянно использовать "На самом деле";
- постоянно говорить о мозге;
- постоянно противопоставлять "ты" и "твой мозг";
- постоянно использовать философские вопросы;
- постоянно заканчивать "Напиши в комментариях";
- повторять одну мысль разными словами;
- использовать одинаковый тип hook в каждом видео.

Каждая сцена должна добавлять новую информацию.

==================================================
ФОРМАТЫ
==================================================

SCIENTIFIC_SHOCK:

Начни с неожиданного научного факта.
Объясни механизм.
Покажи последствия.
Заверши выводом, который меняет восприятие
явления.


THOUGHT_EXPERIMENT:

Предложи зрителю представить необычную
ситуацию.
Постепенно объясни научные последствия.
Развивай идею от простого к сложному.
Заверши парадоксальным выводом.


REAL_EXPERIMENT:

Расскажи о реальном научном эксперименте.
Объясни, что сделали учёные.
Покажи неожиданный результат.
Расскажи, чему эксперимент научил.


MYTH_BUSTING:

Начни с распространённого убеждения.
Покажи, почему оно может быть ошибочным.
Объясни научную реальность.
Заверши неожиданным фактом.


HIDDEN_MECHANISM:

Возьми обычное явление из повседневной жизни.
Покажи скрытый механизм.
Постепенно раскрой его.
Сделай обычное явление необычным.


HISTORICAL_DISCOVERY:

Расскажи историю реального научного открытия.
Покажи проблему, случайность или неожиданное
наблюдение.
Раскрой момент открытия.
Покажи его значение.


FUTURE_SCENARIO:

Опиши реалистичный научный сценарий будущего.
Покажи, как открытие или технология может изменить
жизнь.
Раскрой неожиданные последствия.


DARK_QUESTION:

Начни с тревожной научной загадки.
Постепенно раскрой известные факты.
Не давай простого ответа.
Оставь сильную мысль в финале.


==================================================
НАУЧНАЯ ДОСТОВЕРНОСТЬ
==================================================

Используй реальные научные факты.

Не выдумывай:

- исследования;
- эксперименты;
- учёных;
- исторические события;
- научные открытия.

Если факт часто пересказывается упрощённо,
используй осторожную формулировку.

Не выдавай популярную легенду за доказанный факт.

==================================================
SEARCH
==================================================

SEARCH должен быть только для Pexels.

Используй реальные визуальные объекты,
людей, места, животных, лаборатории и природные
явления, которые реально можно найти на Pexels.

ХОРОШО:

space galaxy stars telescope

scientist laboratory microscope

laboratory Petri dish bacteria science

ocean waves storm

volcano eruption nature

animals nature documentary

human eye macro close up

robot artificial intelligence computer

old scientific notes handwriting paper

medical laboratory glass flasks

clock time lapse

forest wildlife nature

underwater deep ocean

ПОПАДАЕТСЯ ПЛОХО:

alien hologram

magic portal

abstract energy

mysterious cosmic power

digital consciousness

invisible force

quantum energy visualisation

SEARCH должен описывать конкретный
визуальный объект или действие.

==================================================
СЦЕНАРИЙ
==================================================

- ровно 10 сцен;
- около 60 секунд;
- первая сцена — сильный hook;
- каждая следующая сцена развивает историю;
- каждая сцена содержит новую информацию;
- финал должен соответствовать выбранному формату;
- призыв к комментариям использовать только если
  он действительно естественен.

==================================================
ФОРМАТ ОТВЕТА
==================================================

TITLE:

название


TOPIC:

конкретная тема ролика


SCENE 1:

TEXT:

текст

SEARCH:

запрос


SCENE 2:

TEXT:

текст

SEARCH:

запрос


SCENE 3:

TEXT:

текст

SEARCH:

запрос


SCENE 4:

TEXT:

текст

SEARCH:

запрос


SCENE 5:

TEXT:

текст

SEARCH:

запрос


SCENE 6:

TEXT:

текст

SEARCH:

запрос


SCENE 7:

TEXT:

текст

SEARCH:

запрос


SCENE 8:

TEXT:

текст

SEARCH:

запрос


SCENE 9:

TEXT:

текст

SEARCH:

запрос


SCENE 10:

TEXT:

текст

SEARCH:

запрос


Не добавляй никаких пояснений
до или после сценария.
"""


    # =====================================================
    # GENERATION
    # =====================================================

    for model in MODELS:


        try:


            print(

                f"Пробуем модель: {model}"

            )


            response = client.models.generate_content(

                model=model,

                contents=prompt

            )


            text = response.text


            print(

                "\n📄 Получен сценарий:\n"

            )


            print(

                text

            )


            # ---------------------------------------------
            # Сохраняем историю только после успешной
            # генерации
            # ---------------------------------------------

            title = extract_title(

                text

            )


            topic = extract_topic(

                text

            )


            save_history(

                title=title,

                category=topic_category,

                script_format=script_format,

                emotion=emotion,

                topic=topic

            )


            return text


        except Exception as e:


            print(

                f"Ошибка модели {model}: {e}"

            )


    raise Exception(

        "Нет доступных моделей"

    )
