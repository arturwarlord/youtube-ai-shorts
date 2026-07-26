import gc

import whisper


# ==========================
# CONFIG
# ==========================

MODEL_NAME = "tiny"


# ==========================
# GLOBAL MODEL CACHE
# ==========================

MODEL = None


# ==========================
# LOAD MODEL
# ==========================

def get_model():

    global MODEL


    if MODEL is None:

        print(
            "🧠 Загрузка Whisper модели..."
        )


        # Освобождаем неиспользуемую память
        gc.collect()


        MODEL = whisper.load_model(

            MODEL_NAME

        )


        print(

            f"✅ Whisper модель загружена: "
            f"{MODEL_NAME}"

        )


    return MODEL


# ==========================
# TRANSCRIBE AUDIO
# ==========================

def transcribe_audio(

        audio_file

):

    model = get_model()


    print(

        "🎧 Распознавание голоса..."

    )


    result = model.transcribe(

        audio_file,

        language="ru",

        word_timestamps=True,

        fp16=False,

        temperature=0

    )


    words = []


    for segment in result.get(

        "segments",

        []

    ):


        for item in segment.get(

            "words",

            []

        ):


            word = item.get(

                "word",

                ""

            ).strip()


            start = item.get(

                "start",

                None

            )


            end = item.get(

                "end",

                None

            )


            if not word:

                continue


            if start is None or end is None:

                continue


            words.append(

                {

                    "word": word,

                    "start": float(start),

                    "end": float(end)

                }

            )


    print(

        f"✅ Найдено слов: "
        f"{len(words)}"

    )


    if words:

        print(

            "🔤 Пример:"

        )


        print(

            words[:5]

        )


    return words
