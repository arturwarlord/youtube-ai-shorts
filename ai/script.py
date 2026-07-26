import os
import json
import random
import re

from dotenv import load_dotenv
from google import genai


# =========================================================
# ENV
# =========================================================

load_dotenv()


# =========================================================
# GEMINI
# =========================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_KEY")
)


MODELS = [

    "gemini-flash-lite-latest",

    "gemini-2.0-flash",

    "gemini-2.0-flash-lite"

]


# =========================================================
# HISTORY
# =========================================================

HISTORY_FILE = "topic_history.json"


MAX_HISTORY = 50


# =========================================================
# CONTENT SETTINGS
# =========================================================

CATEGORIES = [

    "мозг и сознание",

    "космос и Вселенная",

    "история научных открытий",

    "научные эксперименты",

    "человеческое тело",

    "животные и природа",

    "океан и глубины",

    "физика повседневности",

    "технологии",

    "психология",

    "парадоксы науки",

    "загадки прошлого",

    "медицина",

    "эволюция",

    "катастрофы и выживание"

]


FORMATS = [

    "HISTORICAL_DISCOVERY",

    "SCIENTIFIC_MYSTERY",

    "EXPERIMENT",

    "UNEXPECTED_FACT",

    "PARADOX",

    "HIDDEN_HISTORY",

    "SCIENTIFIC_FAILURE",

    "HUMAN_LIMITS",

    "NATURE_MYSTERY",

    "FUTURE_PROBLEM"

]


EMOTIONS = [

    "шок",

    "научное любопытство",

    "тревога",

    "удивление",

    "восхищение",

    "страх",

    "парадокс",

    "тайна"

]


# =========================================================
# HISTORY FUNCTIONS
# =========================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):

        return []


    try:

        with open(

            HISTORY_FILE,

            "r",

            encoding="utf-8"

        ) as file:

            data = json.load(file)


        if isinstance(data, list):

            return data


        return []


    except Exception as e:

        print(
            f"⚠️ Ошибка чтения истории: {e}"
        )

        return []



def save_history(topic_data):

    history = load_history()


    history.append(topic_data)


    history = history[-MAX_HISTORY:]


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

                indent=4

            )


        print(
            "💾 История темы сохранена"
        )


    except Exception as e:

        print(
            f"⚠️ Ошибка сохранения истории: {e}"
        )



def get_history_context():

    history = load_history()


    if not history:

        return "История предыдущих тем пуста."


    result = []


    for item in history[-30:]:

        title = item.get(

            "title",

            ""

        )


        topic = item.get(

            "topic",

            ""

        )


        core_idea = item.get(

            "core_idea",

            ""

        )


        entities = item.get(

            "used_entities",

            []

        )


        result.append(

            f"- TITLE: {title}\n"
            f"  TOPIC: {topic}\n"
            f"  CORE IDEA: {core_idea}\n"
            f"  ENTITIES: {', '.join(entities)}"

        )


    return "\n".join(result)


# =========================================================
# WORD COUNT
# =========================================================

def count_script_words(text):

    scenes = re.findall(

        r"TEXT:\s*(.*?)(?=\n\s*SEARCH:)",

        text,

        re.DOTALL | re.IGNORECASE

    )


    full_text = " ".join(scenes)


    words = re.findall(

        r"\b[\wЁёА-я-]+\b",

        full_text

    )


    return len(words)



def count_scenes(text):

    scenes = re.findall(

        r"SCENE\s+\d+\s*:",

        text,

        re.IGNORECASE

    )


    return len(scenes)


# =========================================================
# EXTRACT SCENES
# =========================================================

def extract_scenes(text):

    pattern = (

        r"SCENE\s+(\d+)\s*:\s*"

        r"TEXT:\s*(.*?)\s*"

        r"SEARCH:\s*(.*?)(?="

        r"\n\s*SCENE\s+\d+\s*:"

        r"|\Z)"

    )


    return re.findall(

        pattern,

        text,

        re.DOTALL | re.IGNORECASE

    )


# =========================================================
# SCRIPT VALIDATION
# =========================================================

def validate_script(text):

    errors = []


    # =========================================
    # BASIC FIELDS
    # =========================================

    required_fields = [

        "TITLE:",

        "TOPIC:",

        "CORE IDEA:",

        "USED ENTITIES:",

        "SCENE 1:",

        "SCENE 10:"

    ]


    for field in required_fields:

        if field not in text:

            errors.append(

                f"Отсутствует поле: {field}"

            )


    # =========================================
    # SCENES
    # =========================================

    scenes = extract_scenes(text)


    scene_count = len(scenes)


    if scene_count != 10:

        errors.append(

            f"Найдено сцен: {scene_count}, "
            f"нужно ровно 10"

        )


    # =========================================
    # WORD COUNT
    # =========================================

    word_count = count_script_words(text)


    # Минимум снижен с 100 до 85.
    #
    # 85–130 слов подходят
    # для короткого динамичного Shorts.
    #

    if word_count < 85:

        errors.append(

            f"Слишком короткий сценарий: "
            f"{word_count} слов"

        )


    if word_count > 130:

        errors.append(

            f"Слишком длинный сценарий: "
            f"{word_count} слов"

        )


    # =========================================
    # SCENE CONTENT VALIDATION
    # =========================================

    for scene_number, scene_text, search_query in scenes:


        scene_text = scene_text.strip()

        search_query = search_query.strip()


        if not scene_text:

            errors.append(

                f"SCENE {scene_number}: "
                f"пустой TEXT"

            )


        if not search_query:

            errors.append(

                f"SCENE {scene_number}: "
                f"пустой SEARCH"

            )


    # =========================================
    # USED ENTITIES VALIDATION
    # =========================================

    entities_match = re.search(

        r"USED ENTITIES:\s*(.*?)(?=\n\s*SCENE\s+1\s*:)",

        text,

        re.DOTALL | re.IGNORECASE

    )


    if entities_match:

        entities_text = (

            entities_match.group(1)
            .strip()

        )


        if not entities_text:

            errors.append(

                "USED ENTITIES пустой"

            )


    # =========================================
    # FORBIDDEN PHRASES
    # =========================================

    forbidden_phrases = [

        "ты когда-нибудь задумывался",

        "ты думаешь",

        "представь себе",

        "мало кто знает",

        "учёные доказали",

        "это чистая случайность"

    ]


    text_lower = text.lower()


    for phrase in forbidden_phrases:

        if phrase in text_lower:

            errors.append(

                f"Шаблонная фраза: {phrase}"

            )


    # =========================================
    # RESULT
    # =========================================

    if errors:

        print(

            "\n⚠️ Проверка сценария "
            "не пройдена:"

        )


        for error in errors:

            print(

                f"   ❌ {error}"

            )


        return False


    print(

        f"✅ Проверка пройдена: "

        f"{scene_count} сцен, "

        f"{word_count} слов"

    )


    return True


# =========================================================
# EXTRACT FIELD
# =========================================================

def extract_field(

    text,

    field_name

):

    pattern = (

        rf"{field_name}:\s*(.*?)(?=\n[A-ZА-Я ]+:|\Z)"

    )


    match = re.search(

        pattern,

        text,

        re.DOTALL | re.IGNORECASE

    )


    if not match:

        return ""


    return match.group(1).strip()


# =========================================================
# EXTRACT TOPIC DATA
# =========================================================

def extract_topic_data(

    text,

    category,

    content_format,

    emotion

):

    title = extract_field(

        text,

        "TITLE"

    )


    topic = extract_field(

        text,

        "TOPIC"

    )


    core_idea = extract_field(

        text,

        "CORE IDEA"

    )


    entities_text = extract_field(

        text,

        "USED ENTITIES"

    )


    if not core_idea:

        core_idea = topic


    if entities_text:

        entities = [

            item.strip()

            for item in re.split(

                r",|;",

                entities_text

            )

            if item.strip()

        ]

    else:

        entities = []


    return {

        "title": title,

        "topic": topic,

        "core_idea": core_idea,

        "used_entities": entities,

        "category": category,

        "format": content_format,

        "emotion": emotion

    }


# =========================================================
# PROMPT
# =========================================================

def build_prompt(

    category,

    content_format,

    emotion,

    history_context

):

    return f"""

Ты профессиональный сценарист
научного YouTube Shorts-канала
уровня Dark Science.

Твоя задача — создать ОДИН уникальный,
научно-популярный сценарий.

==================================================
ПАРАМЕТРЫ ЭТОГО РОЛИКА
==================================================

КАТЕГОРИЯ:
{category}

ФОРМАТ:
{content_format}

ГЛАВНАЯ ЭМОЦИЯ:
{emotion}

==================================================
ИСТОРИЯ ПРЕДЫДУЩИХ ТЕМ
==================================================

{history_context}

==================================================
ГЛАВНОЕ ПРАВИЛО
==================================================

ВЕСЬ РОЛИК ДОЛЖЕН БЫТЬ ПОСВЯЩЁН
ОДНОЙ КОНКРЕТНОЙ ЦЕНТРАЛЬНОЙ ТЕМЕ.

Не меняй тему после начала.

Все 10 сцен должны объяснять,
развивать и завершать ОДНУ И ТУ ЖЕ
главную идею.

НЕЛЬЗЯ:

- начинать с одного научного объекта,
  а заканчивать другой темой;

- смешивать разные научные гипотезы;

- добавлять случайную тему
  ради драматического эффекта;

- переходить от космоса к инопланетянам,
  если ролик изначально о конкретном
  астрономическом объекте;

- переходить от физического явления
  к психологии без прямой связи;

- добавлять цивилизации, пришельцев
  или заговоры, если они не являются
  частью основной темы.

Если TOPIC:

"квазар J0529-4351"

то все сцены должны быть
непосредственно связаны с:

- этим квазаром;

- его наблюдением;

- его массой;

- его светом;

- его расстоянием;

- его физическими характеристиками;

- научной загадкой вокруг него.

==================================================
ЗАПРЕТ НА ПОВТОРЫ
==================================================

Новая тема НЕ ДОЛЖНА:

- повторять предыдущий ролик;

- быть перефразированной версией старой темы;

- использовать ту же центральную идею;

- повторять тот же научный парадокс;

- повторять главного исторического персонажа;

- повторять основной объект исследования.

==================================================
ДРАМАТУРГИЯ
==================================================

СЦЕНА 1:

Мощный hook.

СЦЕНЫ 2–3:

Контекст и загадка.

СЦЕНЫ 4–6:

Развитие истории,
эксперимент, открытие
или научный конфликт.

СЦЕНЫ 7–8:

Самый неожиданный научный факт.

СЦЕНА 9:

Главный вывод или переворот.

СЦЕНА 10:

Сильное завершение,
которое вызывает комментарии.

==================================================
HOOK
==================================================

Первая сцена должна сразу
создать вопрос или конфликт.

НЕ начинай автоматически с:

"В таком-то году..."

"Учёные обнаружили..."

"Ты когда-нибудь задумывался..."

"Ты думаешь..."

"Представь себе..."

"Мало кто знает..."

Предпочитай:

- неожиданный результат;

- парадокс;

- научную ошибку;

- странное наблюдение;

- опасное последствие;

- вопрос без очевидного ответа.

==================================================
ДЛИНА
==================================================

Ровно 10 сцен.

Общий текст всех сцен:

90–125 русских слов.

Целевая длина:

95–120 слов.

Абсолютный максимум:

130 слов.

Целевая длительность озвучки:

50–65 секунд.

Каждая сцена:

примерно 9–14 слов.

Используй короткие,
динамичные предложения.

==================================================
НАУЧНАЯ ТОЧНОСТЬ
==================================================

Не превращай популярную легенду
в доказанный научный факт.

Особенно проверяй:

- даты;

- имена;

- причинно-следственные связи;

- научные открытия;

- эксперименты;

- исторические события.

Не используй абсолютные формулировки,
если они могут быть неточными.

Избегай:

"учёные доказали, что..."

"все расчёты были абсолютно неверными"

"это была чистая случайность"

если это не является бесспорным фактом.

Используй:

"считалось, что..."

"учёные предполагали..."

"позже выяснилось..."

"одна из гипотез..."

"позже стало понятно..."

==================================================
SEARCH
==================================================

SEARCH используется только для Pexels.

Используй реальные,
визуально доступные запросы.

Каждый SEARCH должен описывать:

- реальный объект;

- реальное место;

- реальное действие;

- реальный природный процесс;

- реальный научный эксперимент.

Пиши SEARCH только на английском языке.

Избегай:

- abstract energy;

- invisible force;

- quantum consciousness;

- magic portal;

- alien hologram;

- future technology concept.

==================================================
USED ENTITIES
==================================================

Поле USED ENTITIES ОБЯЗАТЕЛЬНО.

Оно должно содержать
минимум 3 конкретных элемента,
связанных с основной темой.

Например:

электроны, атомы,
принцип исключения Паули

или:

квазар J0529-4351,
сверхмассивная чёрная дыра,
аккреционный диск

==================================================
ФОРМАТ ОТВЕТА
==================================================

TITLE:

название

TOPIC:

одна конкретная тема

CORE IDEA:

главная идея ролика
одним предложением

USED ENTITIES:

минимум 3 важных элемента,
через запятую

SCENE 1:

TEXT:

текст

SEARCH:

реальный Pexels запрос

SCENE 2:

TEXT:

текст

SEARCH:

реальный Pexels запрос

...

SCENE 10:

TEXT:

текст

SEARCH:

реальный Pexels запрос

==================================================
ФИНАЛЬНАЯ ПРОВЕРКА ПЕРЕД ОТВЕТОМ
==================================================

Перед отправкой проверь:

1. Ровно 10 сцен.

2. Общий текст 90–125 слов.

3. Каждая сцена имеет TEXT.

4. Каждая сцена имеет SEARCH.

5. USED ENTITIES содержит минимум 3 элемента.

6. Все сцены относятся к одной центральной теме.

7. СЦЕНА 10 завершает именно эту тему.

8. Нет запрещённых шаблонных фраз.

9. SEARCH написан на английском языке.

10. Не добавляй пояснения.

Верни только сценарий.
"""


# =========================================================
# GENERATE SCRIPT
# =========================================================

def create_script():

    print(

        "🤖 Создание сценария..."

    )


    category = random.choice(

        CATEGORIES

    )


    content_format = random.choice(

        FORMATS

    )


    emotion = random.choice(

        EMOTIONS

    )


    print(

        f"🎭 Формат: {content_format}"

    )


    print(

        f"🌍 Категория: {category}"

    )


    print(

        f"🎯 Эмоция: {emotion}"

    )


    history_context = get_history_context()


    prompt = build_prompt(

        category,

        content_format,

        emotion,

        history_context

    )


    max_attempts = 3


    for attempt in range(

        max_attempts

    ):


        print(

            f"\n🔄 Попытка генерации "

            f"{attempt + 1}/{max_attempts}"

        )


        for model in MODELS:


            try:

                print(

                    f"Пробуем модель: {model}"

                )


                response = client.models.generate_content(

                    model=model,

                    contents=prompt

                )


                text = response.text.strip()


                print(

                    "\n📄 Получен сценарий:\n"

                )


                print(text)


                print(

                    "\n🔍 Проверка сценария..."

                )


                if validate_script(text):


                    topic_data = extract_topic_data(

                        text,

                        category,

                        content_format,

                        emotion

                    )


                    save_history(

                        topic_data

                    )


                    return text


                print(

                    "\n⚠️ Сценарий "
                    "не прошёл проверку."

                )


                print(

                    "🔁 Генерируем "
                    "новый вариант..."

                )


                break


            except Exception as e:

                print(

                    f"Ошибка модели "
                    f"{model}: {e}"

                )


    raise Exception(

        "❌ Не удалось создать "

        "корректный сценарий"

    )
