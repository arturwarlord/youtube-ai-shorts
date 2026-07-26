import os
import json
import re

from google import genai
from dotenv import load_dotenv


load_dotenv()


# ==========================================
# CONFIG
# ==========================================

GEMINI_KEY = os.getenv("GEMINI_KEY")


client = genai.Client(
    api_key=GEMINI_KEY
)


MODELS = [
    "gemini-flash-lite-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]


# ==========================================
# CLEAN JSON
# ==========================================

def clean_json(text):

    text = text.strip()

    # Убираем markdown-блоки
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    # Иногда Gemini добавляет текст до JSON
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        text = text[start:end + 1]

    return text


# ==========================================
# GENERATE METADATA
# ==========================================

def generate_metadata(script):

    print("\n🧠 Генерация YouTube metadata...")

    prompt = f"""
Ты профессиональный YouTube Shorts SEO-редактор.

Проанализируй следующий сценарий:

--------------------
{script}
--------------------

Создай metadata для YouTube Shorts.

ТРЕБОВАНИЯ:

1. TITLE

Создай короткий, мощный и кликабельный заголовок.

Правила:

- язык: русский
- максимум 100 символов
- не используй дешёвый кликбейт
- заголовок должен вызывать любопытство
- не раскрывай весь ответ
- не используй кавычки
- без эмодзи

2. DESCRIPTION

Создай описание на русском языке.

Правила:

- 2–4 коротких абзаца
- естественный язык
- кратко раскрывает тему
- вызывает желание досмотреть видео
- не используй фразы вроде "в этом видео"
- не используй чрезмерный кликбейт

3. HASHTAGS

Создай 4–7 релевантных хештегов.

Обязательно:

- #shorts

Остальные хештеги должны быть связаны с темой.

4. TAGS

Создай 8–15 поисковых тегов.

Теги должны быть:

- на русском языке
- связаны с темой
- реальными поисковыми запросами
- без символа #

Верни ТОЛЬКО JSON:

{{
    "title": "...",
    "description": "...",
    "hashtags": [
        "#shorts",
        "#..."
    ],
    "tags": [
        "...",
        "..."
    ]
}}
"""

    for model in MODELS:

        try:

            print(
                f"🤖 Metadata model: {model}"
            )

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            raw_text = response.text

            clean_text = clean_json(
                raw_text
            )

            metadata = json.loads(
                clean_text
            )

            # ==================================
            # VALIDATION
            # ==================================

            if not metadata.get("title"):

                raise ValueError(
                    "Отсутствует title"
                )

            if not metadata.get("description"):

                raise ValueError(
                    "Отсутствует description"
                )

            if not metadata.get("hashtags"):

                raise ValueError(
                    "Отсутствуют hashtags"
                )

            if not metadata.get("tags"):

                raise ValueError(
                    "Отсутствуют tags"
                )

            # ==================================
            # FORCE #SHORTS
            # ==================================

            hashtags = metadata["hashtags"]

            if "#shorts" not in [
                tag.lower()
                for tag in hashtags
            ]:

                hashtags.insert(
                    0,
                    "#shorts"
                )

            metadata["hashtags"] = hashtags

            # ==================================
            # PRINT RESULT
            # ==================================

            print("\n📊 METADATA:")

            print(
                f"\n🎬 TITLE:\n"
                f"{metadata['title']}"
            )

            print(
                f"\n📝 DESCRIPTION:\n"
                f"{metadata['description']}"
            )

            print(
                f"\n#️⃣ HASHTAGS:\n"
                f"{' '.join(metadata['hashtags'])}"
            )

            print(
                f"\n🏷️ TAGS:\n"
                f"{', '.join(metadata['tags'])}"
            )

            return metadata

        except Exception as e:

            print(
                f"⚠️ Ошибка metadata "
                f"({model}): {e}"
            )

    raise RuntimeError(
        "❌ Не удалось создать YouTube metadata"
    )
