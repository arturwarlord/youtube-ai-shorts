```python
import whisper


MODEL_NAME = "tiny"

_model = None


def get_model():
    global _model

    if _model is None:
        print(f"🎙 Loading Whisper model: {MODEL_NAME}")
        _model = whisper.load_model(MODEL_NAME)

    return _model


def transcribe_audio(audio_file):
    """
    Transcribe audio with automatic language detection.

    Returns a list of words with timestamps:

    [
        {
            "word": "Привет",
            "start": 0.12,
            "end": 0.48
        }
    ]
    """

    model = get_model()

    print(f"🎙 Transcribing: {audio_file}")

    result = model.transcribe(
        audio_file,
        language=None,
        word_timestamps=True,
        fp16=False,
        temperature=0,
    )

    words = []

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = word.get("word", "").strip()

            if not text:
                continue

            words.append(
                {
                    "word": text,
                    "start": float(word.get("start", 0)),
                    "end": float(word.get("end", 0)),
                }
            )

    print(f"✅ Transcription complete")
    print(f"📝 Words: {len(words)}")
    print(f"🌍 Language: {result.get('language', 'unknown')}")

    return words


def transcribe_with_language(audio_file):
    """
    Extended transcription.

    Returns both detected language and word timestamps.
    Useful for the new long-video → Shorts pipeline.
    """

    model = get_model()

    print(f"🎙 Transcribing: {audio_file}")

    result = model.transcribe(
        audio_file,
        language=None,
        word_timestamps=True,
        fp16=False,
        temperature=0,
    )

    words = []

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = word.get("word", "").strip()

            if not text:
                continue

            words.append(
                {
                    "word": text,
                    "start": float(word.get("start", 0)),
                    "end": float(word.get("end", 0)),
                }
            )

    language = result.get("language", "unknown")

    print(f"✅ Transcription complete")
    print(f"🌍 Detected language: {language}")
    print(f"📝 Words: {len(words)}")

    return {
        "language": language,
        "words": words,
    }
```
