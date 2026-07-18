import whisper


MODEL = None



def get_model():

    global MODEL


    if MODEL is None:

        print("🧠 Загрузка Whisper модели...")

        MODEL = whisper.load_model(
            "base"
        )


    return MODEL





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
        word_timestamps=True
    )


    words = []


    for segment in result["segments"]:

        for word in segment["words"]:

            words.append({

                "word":
                    word["word"].strip(),

                "start":
                    word["start"],

                "end":
                    word["end"]

            })


    print(
        f"✅ Найдено слов: {len(words)}"
    )


    return words
