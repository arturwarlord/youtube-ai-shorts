import os
import sys
import json
import time
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import yt_dlp

from video.source import (
    validate_source_video,
    get_duration,
    extract_audio,
)

from audio.whisper import transcribe_with_language
from ai.clip_selector import select_clips
from ai.metadata import generate_metadata
from video.thumbnails import create_thumbnail

from video.clip_renderer import render_clips
from video.subtitles import (
    create_ass_subtitles,
    burn_subtitles,
)

from youtube.upload import upload_video


# ============================================================
# CONFIG
# ============================================================

SOURCE_VIDEO = Path("input/source.mp4")
AUDIO_FILE = Path("output/source_audio.wav")
CLIPS_DIR = Path("output/clips")

RUTUBE_CHANNEL_URL = "https://rutube.ru/channel/23968031/"

# None = весь канал.
# extract_flat получает только метаданные,
# поэтому сами видео здесь НЕ скачиваются.
RUTUBE_MAX_VIDEOS = None

# Минимальная длина исходного видео.
# Всё, что короче, не скачиваем.
MIN_SOURCE_DURATION = 20 * 60

# Сколько Shorts публикуем за один запуск.
MAX_CLIPS = 2

MAX_VIDEO_HEIGHT = 720

DOWNLOAD_TIMEOUT = 25 * 60

RETRY_DELAY = 3


# ============================================================
# CLIP CONFIG
# ============================================================

# Gemini иногда возвращает, например:
#
# 40.1 -> 53.3 = 13.2 сек
#
# Для Shorts нам нужен минимум 20 секунд.
#
# Если Gemini дал короткий отрезок, main.py расширяет его
# вокруг исходного момента.
MIN_CLIP_DURATION = 20.0
MAX_CLIP_DURATION = 60.0

# Если новый кандидат пересекается с уже опубликованным
# участком более чем на этот процент, считаем его дублем.
#
# Например:
#
# Published: 100 -> 140
# New:       110 -> 145
#
# Пересечение = 30 сек.
# Новый ролик длиной 35 сек.
# 30 / 35 = 85.7%
#
# => дубль.
DUPLICATE_OVERLAP_RATIO = 0.50

# Если Gemini снова предлагает уже опубликованные фрагменты,
# делаем несколько новых попыток выбора.
#
# Это позволяет искать новые моменты в старом видео.
MAX_CLIP_SELECTION_ATTEMPTS = 3


# ============================================================
# GEMINI RETRY
# ============================================================

GEMINI_MAX_RETRIES = 10

GEMINI_RETRY_BUFFER = 1.5

GEMINI_DEFAULT_RETRY_DELAY = 60


# ============================================================
# PROCESSING HISTORY
# ============================================================

# ВАЖНО:
#
# Здесь теперь хранится НЕ просто:
#
# video_id -> processed
#
# а:
#
# video_id -> все уже опубликованные клипы этого видео.
#
# Поэтому один RUTUBE-ролик можно обрабатывать
# много раз и искать в нём новые моменты.
HISTORY_FILE = Path("data/processed_videos.json")

SOURCE_INFO_FILE = Path("input/source.json")

SELECTED_SOURCE_VIDEO_INFO = {}


# ============================================================
# HISTORY
# ============================================================

def load_processing_history():
    """
    Загружает постоянную историю опубликованных клипов.
    """

    if not HISTORY_FILE.exists():
        return {"videos": {}}

    try:
        data = json.loads(
            HISTORY_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as error:

        print(
            f"⚠️ Could not read {HISTORY_FILE}: {error}"
        )

        return {
            "videos": {}
        }

    if not isinstance(data, dict):
        return {
            "videos": {}
        }

    if not isinstance(
        data.get("videos"),
        dict,
    ):
        data["videos"] = {}

    return data


def save_processing_history(history):
    """
    Безопасно сохраняет историю через временный файл.
    """

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = HISTORY_FILE.with_suffix(
        ".tmp"
    )

    temp_file.write_text(
        json.dumps(
            history,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temp_file.replace(
        HISTORY_FILE
    )


def get_video_history(video_id):
    """
    Возвращает историю конкретного RUTUBE-видео.
    """

    video_id = str(
        video_id or ""
    ).strip()

    if not video_id:
        return {}

    history = load_processing_history()

    videos = history.get(
        "videos",
        {},
    )

    item = videos.get(
        video_id
    )

    if not isinstance(
        item,
        dict,
    ):
        return {}

    return item


def get_published_clips(video_id):
    """
    Возвращает список уже опубликованных клипов
    конкретного исходного видео.
    """

    item = get_video_history(
        video_id
    )

    clips = item.get(
        "clips",
        [],
    )

    if not isinstance(
        clips,
        list,
    ):
        return []

    result = []

    for clip in clips:

        if not isinstance(
            clip,
            dict,
        ):
            continue

        try:

            start = float(
                clip["start"]
            )

            end = float(
                clip["end"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if end <= start:
            continue

        result.append(
            {
                **clip,
                "start": start,
                "end": end,
            }
        )

    return result


def record_published_clip(
    video,
    clip,
    youtube_video_id,
):
    """
    Добавляет ОДИН успешно опубликованный Shorts
    в постоянную историю.

    ВАЖНО:
    существующие клипы НЕ удаляются.
    """

    video_id = str(
        video.get("id") or ""
    ).strip()

    if not video_id:
        print(
            "⚠️ Cannot save published clip: "
            "source video ID is missing."
        )

        return False

    try:

        start = float(
            clip["start"]
        )

        end = float(
            clip["end"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        print(
            "⚠️ Cannot save published clip: "
            "invalid clip timestamps."
        )

        return False

    history = load_processing_history()

    videos = history.setdefault(
        "videos",
        {},
    )

    video_history = videos.get(
        video_id
    )

    if not isinstance(
        video_history,
        dict,
    ):
        video_history = {
            "id": video_id,
            "title": video.get("title") or "",
            "webpage_url": video.get("webpage_url") or "",
            "upload_date": video.get("upload_date") or "",
            "clips": [],
        }

    clips = video_history.get(
        "clips"
    )

    if not isinstance(
        clips,
        list,
    ):
        clips = []

    # --------------------------------------------------------
    # Защита от повторной записи одного и того же клипа
    # --------------------------------------------------------

    for existing in clips:

        if not isinstance(
            existing,
            dict,
        ):
            continue

        try:

            existing_start = float(
                existing["start"]
            )

            existing_end = float(
                existing["end"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        overlap = calculate_overlap_ratio(
            start,
            end,
            existing_start,
            existing_end,
        )

        if overlap >= DUPLICATE_OVERLAP_RATIO:

            print()
            print(
                "⚠️ Clip already exists in history."
            )

            print(
                f"   Existing: "
                f"{existing_start:.2f} → "
                f"{existing_end:.2f}"
            )

            print(
                f"   New:      "
                f"{start:.2f} → "
                f"{end:.2f}"
            )

            return True

    clip_record = {
        "start": round(
            start,
            3,
        ),
        "end": round(
            end,
            3,
        ),
        "duration": round(
            end - start,
            3,
        ),
        "youtube_video_id": (
            str(
                youtube_video_id
            ).strip()
            if youtube_video_id
            else None
        ),
        "published_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    clips.append(
        clip_record
    )

    video_history["id"] = video_id

    video_history["title"] = (
        video.get("title")
        or video_history.get("title")
        or ""
    )

    video_history["webpage_url"] = (
        video.get("webpage_url")
        or video_history.get("webpage_url")
        or ""
    )

    video_history["upload_date"] = (
        video.get("upload_date")
        or video_history.get("upload_date")
        or ""
    )

    video_history["last_processed_at"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    video_history["clips"] = clips

    videos[video_id] = video_history

    save_processing_history(
        history
    )

    print()
    print(
        "💾 Published clip saved to history"
    )

    print(
        f"   Source ID: {video_id}"
    )

    print(
        f"   Clip:      {start:.2f} → {end:.2f}"
    )

    if youtube_video_id:
        print(
            f"   YouTube:   {youtube_video_id}"
        )

    print(
        f"   History:   {HISTORY_FILE}"
    )

    return True


def persist_history_to_git():
    """
    Commit/push history в GitHub Actions.

    Требует:
        permissions:
          contents: write
    """

    if not os.environ.get(
        "GITHUB_ACTIONS"
    ):

        print(
            "ℹ️ Local run: "
            "processing history saved locally."
        )

        return True

    try:

        subprocess.run(
            [
                "git",
                "config",
                "user.name",
                "github-actions[bot]",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        subprocess.run(
            [
                "git",
                "config",
                "user.email",
                "41898282+github-actions[bot]@users.noreply.github.com",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        subprocess.run(
            [
                "git",
                "add",
                str(HISTORY_FILE),
            ],
            check=True,
        )

        diff = subprocess.run(
            [
                "git",
                "diff",
                "--cached",
                "--quiet",
                "--",
                str(HISTORY_FILE),
            ],
        )

        if diff.returncode == 0:

            print(
                "ℹ️ No new history changes to push."
            )

            return True

        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "chore: update processed video history",
            ],
            check=True,
        )

        subprocess.run(
            [
                "git",
                "push",
            ],
            check=True,
        )

        print(
            "✅ Processing history pushed to GitHub"
        )

        return True

    except subprocess.CalledProcessError as error:

        print()
        print(
            "⚠️ Could not persist processing "
            "history to GitHub."
        )

        print(
            "Make sure the workflow has:"
        )

        print(
            "permissions:"
        )

        print(
            "  contents: write"
        )

        print(
            f"Git error: {error}"
        )

        return False


# ============================================================
# SOURCE INFO
# ============================================================

def save_source_info(video):
    """
    Сохраняет выбранный источник между --download и --process.
    """

    SOURCE_INFO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SOURCE_INFO_FILE.write_text(
        json.dumps(
            video,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def load_source_info():
    """
    Восстанавливает metadata выбранного источника.
    """

    if not SOURCE_INFO_FILE.exists():
        return {}

    try:

        data = json.loads(
            SOURCE_INFO_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as error:

        print(
            f"⚠️ Could not read "
            f"{SOURCE_INFO_FILE}: {error}"
        )

        return {}

    return (
        data
        if isinstance(data, dict)
        else {}
    )


# ============================================================
# HELPERS
# ============================================================

def parse_duration(value):
    """
    Поддерживает:
        int / float
        HH:MM:SS
        MM:SS
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return float(value)

    value = str(
        value
    ).strip()

    if not value:
        return None

    try:

        return float(value)

    except ValueError:
        pass

    parts = value.split(
        ":"
    )

    try:

        parts = [
            int(x)
            for x in parts
        ]

    except ValueError:

        return None

    if len(parts) == 3:

        hours, minutes, seconds = parts

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    if len(parts) == 2:

        minutes, seconds = parts

        return (
            minutes * 60
            + seconds
        )

    if len(parts) == 1:

        return float(
            parts[0]
        )

    return None


def format_duration(seconds):

    if seconds is None:
        return "unknown"

    seconds = int(
        seconds
    )

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    secs = seconds % 60

    if hours:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def clean_source_file(
    remove_source_info=False
):
    """
    Удаляет старый source.mp4.
    """

    SOURCE_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SOURCE_VIDEO.exists():

        print(
            "🗑 Removing old source.mp4"
        )

        SOURCE_VIDEO.unlink()

    if (
        remove_source_info
        and SOURCE_INFO_FILE.exists()
    ):

        print(
            "🗑 Removing old source.json"
        )

        SOURCE_INFO_FILE.unlink()


def print_video_info(
    video,
    index=None,
):

    title = (
        video.get("title")
        or "Unknown title"
    )

    video_id = (
        video.get("id")
        or "unknown"
    )

    duration = parse_duration(
        video.get("duration")
    )

    url = (
        video.get("webpage_url")
        or video.get("url")
        or ""
    )

    prefix = (
        f"[{index}] "
        if index is not None
        else ""
    )

    print(
        f"{prefix}{title}\n"
        f"    ID: {video_id}\n"
        f"    Duration: {format_duration(duration)}\n"
        f"    URL: {url}"
    )


# ============================================================
# CLIP DUPLICATE DETECTION
# ============================================================

def calculate_overlap_ratio(
    start_a,
    end_a,
    start_b,
    end_b,
):
    """
    Возвращает процент пересечения относительно
    меньшего из двух клипов.

    Это лучше, чем сравнивать timestamps напрямую.

    Например:

        Published: 100 → 130
        New:       105 → 135

    Пересечение = 25 сек.

    Меньший клип = 30 сек.

    Ratio = 83%.
    """

    try:

        start_a = float(start_a)
        end_a = float(end_a)

        start_b = float(start_b)
        end_b = float(end_b)

    except (
        TypeError,
        ValueError,
    ):

        return 0.0

    if end_a <= start_a:
        return 0.0

    if end_b <= start_b:
        return 0.0

    intersection_start = max(
        start_a,
        start_b,
    )

    intersection_end = min(
        end_a,
        end_b,
    )

    if (
        intersection_end
        <= intersection_start
    ):
        return 0.0

    intersection = (
        intersection_end
        - intersection_start
    )

    duration_a = (
        end_a
        - start_a
    )

    duration_b = (
        end_b
        - start_b
    )

    reference_duration = min(
        duration_a,
        duration_b,
    )

    if reference_duration <= 0:
        return 0.0

    return (
        intersection
        / reference_duration
    )


def is_duplicate_clip(
    clip,
    published_clips,
):
    """
    Проверяет, является ли Gemini-клип уже опубликованным.
    """

    if not published_clips:
        return False

    try:

        start = float(
            clip["start"]
        )

        end = float(
            clip["end"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        return True

    for existing in published_clips:

        try:

            existing_start = float(
                existing["start"]
            )

            existing_end = float(
                existing["end"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        ratio = calculate_overlap_ratio(
            start,
            end,
            existing_start,
            existing_end,
        )

        if (
            ratio
            >= DUPLICATE_OVERLAP_RATIO
        ):

            print()
            print(
                "⏭ Duplicate clip detected"
            )

            print(
                f"   New:      "
                f"{start:.2f} → {end:.2f}"
            )

            print(
                f"   Existing: "
                f"{existing_start:.2f} → "
                f"{existing_end:.2f}"
            )

            print(
                f"   Overlap:  "
                f"{ratio * 100:.1f}%"
            )

            return True

    return False


def normalize_clip_duration(
    clip,
    source_duration,
):
    """
    Гарантирует длину клипа 20–60 секунд.

    Это дополнительная защита main.py.

    Даже если старый ai/clip_selector.py
    вернул:

        40.1 → 53.3

    main.py превратит это примерно в:

        40.1 → 60.1

    если позволяет исходное видео.
    """

    try:

        start = float(
            clip["start"]
        )

        end = float(
            clip["end"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        return None

    if end <= start:
        return None

    source_duration = float(
        source_duration
    )

    # --------------------------------------------------------
    # Ограничиваем границы исходным видео
    # --------------------------------------------------------

    start = max(
        0.0,
        start,
    )

    end = min(
        source_duration,
        end,
    )

    if end <= start:
        return None

    duration = (
        end - start
    )

    # --------------------------------------------------------
    # Слишком длинный
    # --------------------------------------------------------

    if duration > MAX_CLIP_DURATION:

        end = (
            start
            + MAX_CLIP_DURATION
        )

        if end > source_duration:

            end = source_duration

            start = max(
                0.0,
                end - MAX_CLIP_DURATION,
            )

    # --------------------------------------------------------
    # Слишком короткий
    # --------------------------------------------------------

    duration = (
        end - start
    )

    if duration < MIN_CLIP_DURATION:

        needed = (
            MIN_CLIP_DURATION
            - duration
        )

        # Сначала расширяем вправо.
        right_space = (
            source_duration
            - end
        )

        expand_right = min(
            needed,
            right_space,
        )

        end += expand_right

        needed -= expand_right

        # Если справа места не хватило,
        # расширяем влево.
        left_space = start

        expand_left = min(
            needed,
            left_space,
        )

        start -= expand_left

        needed -= expand_left

        # Последняя попытка:
        # если видео позволяет, центрируем
        # 20-секундный отрезок.
        if needed > 0:

            center = (
                start + end
            ) / 2.0

            start = max(
                0.0,
                center
                - MIN_CLIP_DURATION / 2,
            )

            end = min(
                source_duration,
                start
                + MIN_CLIP_DURATION,
            )

            if (
                end - start
                < MIN_CLIP_DURATION
            ):

                end = min(
                    source_duration,
                    MIN_CLIP_DURATION,
                )

                start = max(
                    0.0,
                    end
                    - MIN_CLIP_DURATION,
                )

    # --------------------------------------------------------
    # Финальная проверка
    # --------------------------------------------------------

    start = max(
        0.0,
        start,
    )

    end = min(
        source_duration,
        end,
    )

    final_duration = (
        end - start
    )

    if (
        final_duration
        < MIN_CLIP_DURATION
    ):

        return None

    if (
        final_duration
        > MAX_CLIP_DURATION
    ):

        end = (
            start
            + MAX_CLIP_DURATION
        )

        if end > source_duration:

            end = source_duration

            start = max(
                0.0,
                end - MAX_CLIP_DURATION,
            )

    normalized = dict(
        clip
    )

    normalized["start"] = round(
        start,
        3,
    )

    normalized["end"] = round(
        end,
        3,
    )

    normalized["duration"] = round(
        end - start,
        3,
    )

    return normalized


def prepare_new_clips(
    candidates,
    source_duration,
    published_clips,
    max_clips,
):
    """
    Нормализует Gemini-кандидатов,
    удаляет дубли и пересечения.
    """

    result = []

    for index, candidate in enumerate(
        candidates or [],
        start=1,
    ):

        normalized = normalize_clip_duration(
            candidate,
            source_duration,
        )

        if normalized is None:

            print(
                f"⏭ Candidate #{index} "
                f"could not be normalized."
            )

            continue

        old_start = float(
            candidate.get(
                "start",
                0,
            )
        )

        old_end = float(
            candidate.get(
                "end",
                0,
            )
        )

        new_start = float(
            normalized["start"]
        )

        new_end = float(
            normalized["end"]
        )

        if (
            abs(old_start - new_start)
            > 0.01
            or
            abs(old_end - new_end)
            > 0.01
        ):

            print()
            print(
                f"🔧 Candidate #{index} "
                f"duration normalized:"
            )

            print(
                f"   Gemini: "
                f"{old_start:.2f} → "
                f"{old_end:.2f} "
                f"({old_end - old_start:.2f}s)"
            )

            print(
                f"   Final:  "
                f"{new_start:.2f} → "
                f"{new_end:.2f} "
                f"({new_end - new_start:.2f}s)"
            )

        if is_duplicate_clip(
            normalized,
            published_clips,
        ):
            continue

        # ----------------------------------------------------
        # Проверяем дубли между новыми кандидатами
        # ----------------------------------------------------

        duplicate_with_new = False

        for existing_new in result:

            ratio = calculate_overlap_ratio(
                normalized["start"],
                normalized["end"],
                existing_new["start"],
                existing_new["end"],
            )

            if (
                ratio
                >= DUPLICATE_OVERLAP_RATIO
            ):

                duplicate_with_new = True

                print()
                print(
                    "⏭ New candidates overlap "
                    "each other."
                )

                print(
                    f"   Candidate: "
                    f"{normalized['start']:.2f} → "
                    f"{normalized['end']:.2f}"
                )

                print(
                    f"   Existing:  "
                    f"{existing_new['start']:.2f} → "
                    f"{existing_new['end']:.2f}"
                )

                break

        if duplicate_with_new:
            continue

        result.append(
            normalized
        )

        if len(result) >= max_clips:
            break

    return result


# ============================================================
# GEMINI RETRY
# ============================================================

def is_gemini_rate_limit_error(
    error
):

    error_text = str(
        error
    ).lower()

    if "429" in error_text:
        return True

    if "resource_exhausted" in error_text:
        return True

    if "quota exceeded" in error_text:
        return True

    if "generate_content_free_tier" in error_text:
        return True

    return False


def get_gemini_retry_delay(
    error
):

    error_text = str(
        error
    )

    patterns = [
        r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retryDelay[\"']?\s*[:=]\s*[\"']?([0-9]+(?:\.[0-9]+)?)s",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            error_text,
            flags=re.IGNORECASE,
        )

        if match:

            try:

                return float(
                    match.group(1)
                )

            except (
                ValueError,
                TypeError,
            ):
                pass

    return None


def select_clips_with_retry(
    transcript_words,
    max_clips,
):
    """
    Вызывает select_clips() с retry при 429.
    """

    attempt = 0

    while True:

        attempt += 1

        print()

        if attempt == 1:

            print(
                "🤖 Calling Gemini clip selector..."
            )

        else:

            print(
                f"🤖 Retrying Gemini clip selector "
                f"(attempt {attempt}/"
                f"{GEMINI_MAX_RETRIES + 1})..."
            )

        try:

            return select_clips(
                transcript_words,
                max_clips=max_clips,
            )

        except Exception as error:

            if not is_gemini_rate_limit_error(
                error
            ):
                raise

            if (
                attempt
                > GEMINI_MAX_RETRIES
            ):

                print()
                print(
                    "❌ Gemini rate limit "
                    "retry limit exceeded."
                )

                raise

            retry_delay = (
                get_gemini_retry_delay(
                    error
                )
            )

            if retry_delay is None:

                retry_delay = (
                    GEMINI_DEFAULT_RETRY_DELAY
                )

                print()
                print(
                    "⚠️ Gemini returned 429, "
                    "but retry time could not "
                    "be detected."
                )

            else:

                print()
                print(
                    "⚠️ Gemini rate limit reached."
                )

                print(
                    f"⏳ Gemini requested retry "
                    f"in {retry_delay:.3f}s"
                )

            wait_time = (
                retry_delay
                + GEMINI_RETRY_BUFFER
            )

            print(
                f"⏳ Waiting "
                f"{wait_time:.1f}s "
                f"before retry..."
            )

            remaining = wait_time

            while remaining > 0:

                sleep_for = min(
                    10,
                    remaining,
                )

                print(
                    f"   ⏱ {remaining:.1f}s remaining..."
                )

                time.sleep(
                    sleep_for
                )

                remaining -= sleep_for

            print()
            print(
                "🔄 Retrying Gemini request..."
            )


# ============================================================
# RUTUBE
# ============================================================

def get_rutube_videos():

    print("=" * 70)
    print("📺 RUTUBE")
    print("=" * 70)

    print(
        f"Channel: {RUTUBE_CHANNEL_URL}"
    )

    print()

    ydl_opts = {
        "quiet": False,
        "no_warnings": False,
        "extract_flat": True,
        "skip_download": True,
    }

    try:

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            info = ydl.extract_info(
                RUTUBE_CHANNEL_URL,
                download=False,
            )

    except Exception as error:

        print()
        print(
            "❌ Failed to read RUTUBE channel"
        )

        print(
            f"Error: {error}"
        )

        return []

    if not info:

        print(
            "❌ RUTUBE returned no information"
        )

        return []

    entries = (
        info.get("entries")
        or []
    )

    videos = []

    for entry in entries:

        if not entry:
            continue

        video = dict(
            entry
        )

        video_id = video.get(
            "id"
        )

        if not video_id:
            continue

        webpage_url = (
            video.get(
                "webpage_url"
            )
        )

        if not webpage_url:

            webpage_url = (
                f"https://rutube.ru/video/"
                f"{video_id}/"
            )

        video["webpage_url"] = (
            webpage_url
        )

        videos.append(
            video
        )

    print(
        f"Found {len(videos)} channel entries"
    )

    print()

    return videos


def get_video_details(
    video
):

    url = video.get(
        "webpage_url"
    )

    if not url:
        return None

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=False,
            )

        return info

    except Exception as error:

        print(
            f"⚠️ Could not read "
            f"video details: {error}"
        )

        return None


def get_video_sort_date(
    video
):

    upload_date = video.get(
        "upload_date"
    )

    if upload_date:

        try:

            return datetime.strptime(
                upload_date,
                "%Y%m%d",
            ).replace(
                tzinfo=timezone.utc
            )

        except Exception:
            pass

    timestamp = video.get(
        "timestamp"
    )

    if timestamp:

        try:

            return datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            )

        except Exception:
            pass

    return datetime.min.replace(
        tzinfo=timezone.utc
    )


def select_rutube_videos(
    videos
):
    """
    Выбирает длинные RUTUBE-видео.

    ВАЖНО:
    ранее обработанные видео НЕ исключаются.

    Мы специально возвращаем их в очередь,
    чтобы Gemini мог искать новые моменты.
    """

    print("=" * 70)
    print("🔎 FILTERING RUTUBE VIDEOS")
    print("=" * 70)

    history = load_processing_history()

    processed_ids = {
        str(video_id)
        for video_id in history.get(
            "videos",
            {},
        ).keys()
    }

    print(
        f"Sources with history: "
        f"{len(processed_ids)}"
    )

    candidates = []

    for index, original_video in enumerate(
        videos,
        start=1,
    ):

        video = dict(
            original_video
        )

        url = video.get(
            "webpage_url"
        )

        if not url:
            continue

        video_id = str(
            video.get("id") or ""
        ).strip()

        if not video_id:
            continue

        title = (
            video.get("title")
            or ""
        )

        title_lower = title.lower()

        # ----------------------------------------------------
        # НЕ СКАЧИВАЕМ очевидные Shorts
        # ----------------------------------------------------

        if (
            "shorts" in title_lower
            or "short" in title_lower
        ):

            print(
                f"⏭ Possible Shorts: "
                f"{title}"
            )

            continue

        duration = parse_duration(
            video.get("duration")
        )

        # ----------------------------------------------------
        # Если duration нет — только тогда
        # запрашиваем полную информацию.
        # ----------------------------------------------------

        if duration is None:

            print(
                f"Checking metadata: {url}"
            )

            full_info = (
                get_video_details(
                    video
                )
            )

            if not full_info:
                continue

            video = full_info

            duration = parse_duration(
                video.get("duration")
            )

            title = (
                video.get("title")
                or title
            )

            title_lower = title.lower()

            if (
                "shorts" in title_lower
                or "short" in title_lower
            ):

                print(
                    f"⏭ Possible Shorts: "
                    f"{title}"
                )

                continue

        if duration is None:

            print(
                "⚠️ Duration unknown — skipping"
            )

            continue

        # ----------------------------------------------------
        # Жёсткий фильтр коротких исходников
        # ----------------------------------------------------

        if duration < MIN_SOURCE_DURATION:

            print(
                f"⏭ Too short: "
                f"{title or 'Unknown'} "
                f"({format_duration(duration)})"
            )

            continue

        video["_sort_date"] = (
            get_video_sort_date(
                video
            )
        )

        video["_has_history"] = (
            video_id in processed_ids
        )

        video["_published_clips_count"] = (
            len(
                get_published_clips(
                    video_id
                )
            )
        )

        candidates.append(
            video
        )

    # ========================================================
    # СОРТИРОВКА
    #
    # 1. Сначала новые источники, которые
    #    ещё вообще не публиковались.
    #
    # 2. Затем старые источники.
    #
    # Среди старых сначала те,
    # которые обрабатывались давно.
    # ========================================================

    new_videos = [
        video
        for video in candidates
        if not video.get(
            "_has_history"
        )
    ]

    old_videos = [
        video
        for video in candidates
        if video.get(
            "_has_history"
        )
    ]

    new_videos.sort(
        key=lambda video: video.get(
            "_sort_date"
        ),
        reverse=True,
    )

    def old_video_sort_key(
        video
    ):

        video_id = str(
            video.get("id") or ""
        )

        item = history.get(
            "videos",
            {},
        ).get(
            video_id,
            {},
        )

        last_processed = (
            item.get(
                "last_processed_at"
            )
            or item.get(
                "processed_at"
            )
            or ""
        )

        if last_processed:

            try:

                dt = datetime.fromisoformat(
                    last_processed.replace(
                        "Z",
                        "+00:00",
                    )
                )

                return dt

            except Exception:
                pass

        return datetime.min.replace(
            tzinfo=timezone.utc
        )

    old_videos.sort(
        key=old_video_sort_key
    )

    candidates = (
        new_videos
        + old_videos
    )

    print()
    print(
        f"Suitable long videos: "
        f"{len(candidates)}"
    )

    print()
    print(
        f"New sources: "
        f"{len(new_videos)}"
    )

    print(
        f"Sources available "
        f"for re-processing: "
        f"{len(old_videos)}"
    )

    print()

    for index, video in enumerate(
        candidates,
        start=1,
    ):

        video_id = str(
            video.get("id") or ""
        )

        published_count = (
            video.get(
                "_published_clips_count",
                0,
            )
        )

        status = (
            "🆕 NEW"
            if not video.get(
                "_has_history"
            )
            else (
                f"🔄 RECHECK "
                f"({published_count} published)"
            )
        )

        print(
            f"[{index}] {status}"
        )

        print_video_info(
            video
        )

        print()

    return candidates


# ============================================================
# DOWNLOAD
# ============================================================

def download_rutube_video(
    video
):

    url = video.get(
        "webpage_url"
    )

    if not url:

        print(
            "❌ Video URL is missing"
        )

        return False

    title = (
        video.get("title")
        or "Unknown"
    )

    print("=" * 70)
    print("⬇️ DOWNLOADING RUTUBE VIDEO")
    print("=" * 70)

    print(
        f"Title: {title}"
    )

    print(
        f"URL:   {url}"
    )

    print()

    clean_source_file(
        remove_source_info=True
    )

    SOURCE_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ydl_opts = {
        "outtmpl": str(
            SOURCE_VIDEO.with_suffix(
                ".%(ext)s"
            )
        ),

        "format": (
            f"bv*[height<={MAX_VIDEO_HEIGHT}]"
            f"+ba/"
            f"b[height<={MAX_VIDEO_HEIGHT}]"
            f"/b"
        ),

        "merge_output_format": "mp4",

        "ffmpeg_location": "ffmpeg",

        "socket_timeout": 30,

        "retries": 5,

        "fragment_retries": 5,

        "keepvideo": False,

        "quiet": False,

        "no_warnings": False,

        "noplaylist": True,
    }

    try:

        start_time = time.time()

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            ydl.download(
                [url]
            )

        elapsed = (
            time.time()
            - start_time
        )

        print()

        print(
            f"Download finished in "
            f"{elapsed:.1f}s"
        )

    except Exception as error:

        print()
        print(
            "❌ RUTUBE download failed"
        )

        print(
            f"Error: {error}"
        )

        clean_source_file()

        return False

    if SOURCE_VIDEO.exists():

        size_mb = (
            SOURCE_VIDEO.stat().st_size
            / 1024
            / 1024
        )

        print()
        print(
            "✅ source.mp4 created"
        )

        print(
            f"Size: {size_mb:.2f} MB"
        )

        if size_mb < 1:

            print(
                "❌ File is suspiciously small"
            )

            clean_source_file()

            return False

        save_source_info(
            video
        )

        print(
            f"💾 Source metadata saved: "
            f"{SOURCE_INFO_FILE}"
        )

        return True

    possible_files = list(
        SOURCE_VIDEO.parent.glob(
            "source.*"
        )
    )

    possible_files = [
        path
        for path in possible_files
        if (
            path.is_file()
            and path.name != "source.mp4"
            and path.suffix.lower()
            in {
                ".mkv",
                ".webm",
                ".mov",
                ".mp4",
                ".m4v",
            }
        )
    ]

    if possible_files:

        source = max(
            possible_files,
            key=lambda path:
            path.stat().st_size,
        )

        print(
            f"⚠️ Found downloaded file: "
            f"{source}"
        )

        try:

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source),
                    "-c",
                    "copy",
                    str(SOURCE_VIDEO),
                ],
                check=True,
            )

            source.unlink(
                missing_ok=True
            )

        except Exception as error:

            print(
                f"❌ Could not convert "
                f"source to MP4: {error}"
            )

            return False

        if SOURCE_VIDEO.exists():

            print(
                "✅ Converted to "
                "input/source.mp4"
            )

            save_source_info(
                video
            )

            print(
                f"💾 Source metadata saved: "
                f"{SOURCE_INFO_FILE}"
            )

            return True

    print(
        "❌ input/source.mp4 "
        "was not created"
    )

    return False


def find_and_download():

    videos = get_rutube_videos()

    if not videos:

        raise RuntimeError(
            "❌ No videos found "
            "on RUTUBE channel."
        )

    candidates = (
        select_rutube_videos(
            videos
        )
    )

    if not candidates:

        raise RuntimeError(
            "❌ No suitable long "
            "RUTUBE videos found."
        )

    print("=" * 70)
    print("🎯 DOWNLOAD CANDIDATES")
    print("=" * 70)

    # Весь список теперь доступен.
    # Но скачиваем по одному.
    #
    # extract_flat уже получил весь канал,
    # поэтому здесь нет лимита 300.
    candidates_to_try = candidates

    for index, video in enumerate(
        candidates_to_try,
        start=1,
    ):

        print()
        print(
            f"Attempt {index}/"
            f"{len(candidates_to_try)}"
        )

        print_video_info(
            video
        )

        if download_rutube_video(
            video
        ):

            print()
            print(
                "🎉 RUTUBE source ready"
            )

            return True

        if (
            index
            < len(candidates_to_try)
        ):

            print()
            print(
                f"Waiting {RETRY_DELAY}s "
                f"before next video..."
            )

            time.sleep(
                RETRY_DELAY
            )

    raise RuntimeError(
        "❌ None of the RUTUBE "
        "candidates could be downloaded."
    )


# ============================================================
# SELECT NEW CLIPS
# ============================================================

def select_new_clips(
    transcript_words,
    source_duration,
    source_video_info,
):
    """
    Выбирает новые клипы.

    Если Gemini снова возвращает уже опубликованные
    фрагменты — повторяем запрос несколько раз.

    Таким образом старое видео можно постепенно
    разбирать на новые Shorts.
    """

    source_id = str(
        source_video_info.get(
            "id"
        )
        or ""
    ).strip()

    published_clips = (
        get_published_clips(
            source_id
        )
    )

    print()
    print(
        f"📚 Previously published clips "
        f"from this source: "
        f"{len(published_clips)}"
    )

    for index, clip in enumerate(
        published_clips,
        start=1,
    ):

        print(
            f"   #{index}: "
            f"{clip['start']:.2f} → "
            f"{clip['end']:.2f}"
            + (
                f" | YouTube: "
                f"{clip.get('youtube_video_id')}"
                if clip.get(
                    "youtube_video_id"
                )
                else ""
            )
        )

    all_new_clips = []

    for attempt in range(
        1,
        MAX_CLIP_SELECTION_ATTEMPTS + 1,
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"🤖 Gemini selection attempt "
            f"{attempt}/"
            f"{MAX_CLIP_SELECTION_ATTEMPTS}"
        )

        print(
            "=" * 60
        )

        candidates = (
            select_clips_with_retry(
                transcript_words,
                max_clips=MAX_CLIPS,
            )
        )

        if not candidates:

            print(
                "⚠️ Gemini returned no candidates."
            )

            continue

        print()
        print(
            f"Gemini candidates: "
            f"{len(candidates)}"
        )

        new_clips = prepare_new_clips(
            candidates,
            source_duration,
            published_clips,
            MAX_CLIPS - len(all_new_clips),
        )

        for clip in new_clips:

            duplicate = False

            for existing in all_new_clips:

                ratio = calculate_overlap_ratio(
                    clip["start"],
                    clip["end"],
                    existing["start"],
                    existing["end"],
                )

                if (
                    ratio
                    >= DUPLICATE_OVERLAP_RATIO
                ):

                    duplicate = True
                    break

            if not duplicate:

                all_new_clips.append(
                    clip
                )

        if len(all_new_clips) >= MAX_CLIPS:

            break

        print()
        print(
            f"⚠️ Only "
            f"{len(all_new_clips)} new clip(s) "
            f"found."
        )

        print(
            "🔄 Asking Gemini for additional "
            "candidates..."
        )

    return all_new_clips[:MAX_CLIPS]


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video():

    global SELECTED_SOURCE_VIDEO_INFO

    print("=" * 70)
    print("🎬 PROCESS VIDEO")
    print("=" * 70)

    if not SELECTED_SOURCE_VIDEO_INFO:

        SELECTED_SOURCE_VIDEO_INFO = (
            load_source_info()
        )

    if SELECTED_SOURCE_VIDEO_INFO:

        print()

        print(
            "📺 Source: "
            f"{SELECTED_SOURCE_VIDEO_INFO.get('title', 'Unknown')}"
        )

        print(
            "🆔 Source ID: "
            f"{SELECTED_SOURCE_VIDEO_INFO.get('id', 'unknown')}"
        )

    # --------------------------------------------------------
    # 1. Validate
    # --------------------------------------------------------

    if not SOURCE_VIDEO.exists():

        raise FileNotFoundError(
            "❌ input/source.mp4 not found"
        )

    print()

    print(
        "1️⃣ Validating source video..."
    )

    validate_source_video(
        SOURCE_VIDEO
    )

    duration = get_duration(
        SOURCE_VIDEO
    )

    print(
        f"Source duration: "
        f"{format_duration(duration)}"
    )

    # --------------------------------------------------------
    # 2. Audio
    # --------------------------------------------------------

    print()

    print(
        "2️⃣ Extracting audio..."
    )

    AUDIO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extract_audio(
        SOURCE_VIDEO,
        AUDIO_FILE,
    )

    # --------------------------------------------------------
    # 3. Whisper
    # --------------------------------------------------------

    print()

    print(
        "3️⃣ Transcribing with Whisper..."
    )

    transcript = (
        transcribe_with_language(
            AUDIO_FILE,
        )
    )

    if not transcript:

        raise RuntimeError(
            "❌ Whisper returned "
            "empty transcript"
        )

    if not transcript.get(
        "words"
    ):

        raise RuntimeError(
            "❌ Whisper returned "
            "no word timestamps"
        )

    print(
        "✅ Transcript ready"
    )

    print(
        f"📝 Words: "
        f"{len(transcript['words'])}"
    )

    # --------------------------------------------------------
    # 4. Gemini
    # --------------------------------------------------------

    print()

    print(
        "4️⃣ Selecting NEW clips with Gemini..."
    )

    clips = select_new_clips(
        transcript["words"],
        duration,
        SELECTED_SOURCE_VIDEO_INFO,
    )

    if not clips:

        print()
        print(
            "=" * 70
        )

        print(
            "ℹ️ NO NEW CLIPS FOUND"
        )

        print(
            "=" * 70
        )

        print()
        print(
            "Gemini did not find a new "
            "non-duplicate clip in this source."
        )

        print(
            "The source remains available "
            "for future re-processing."
        )

        print(
            "=" * 70
        )

        return

    print()

    print(
        f"🆕 New clips selected: "
        f"{len(clips)}"
    )

    for index, clip in enumerate(
        clips,
        start=1,
    ):

        print()

        print(
            f"Clip {index}:"
        )

        print(
            json.dumps(
                clip,
                ensure_ascii=False,
                indent=2,
            )
        )

    # --------------------------------------------------------
    # 5. Render
    # --------------------------------------------------------

    print()

    print(
        "5️⃣ Rendering Shorts..."
    )

    CLIPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered = render_clips(
        SOURCE_VIDEO,
        clips,
        output_dir=CLIPS_DIR,
    )

    if not rendered:

        raise RuntimeError(
            "❌ No clips were rendered"
        )

    if len(rendered) != len(
        clips
    ):

        raise RuntimeError(
            "❌ Number of rendered "
            "clips does not match "
            "selected clips"
        )

    print()

    print(
        f"✅ Rendered clips: "
        f"{len(rendered)}"
    )

    # --------------------------------------------------------
    # 6 + 7 + 8
    # --------------------------------------------------------

    print()

    print(
        "6️⃣ Adding subtitles..."
    )

    print()

    print(
        "7️⃣ Generating metadata..."
    )

    print()

    print(
        "8️⃣ Generating thumbnails..."
    )

    final_videos = []

    metadata_files = []

    thumbnail_files = []

    uploaded_videos = []

    for index, (
        rendered_path,
        clip,
    ) in enumerate(
        zip(
            rendered,
            clips,
        ),
        start=1,
    ):

        subtitle_path = (
            CLIPS_DIR
            / f"clip_{index:02d}.ass"
        )

        final_path = (
            CLIPS_DIR
            / f"clip_{index:02d}_final.mp4"
        )

        metadata_path = (
            CLIPS_DIR
            / f"clip_{index:02d}_metadata.json"
        )

        thumbnail_path = (
            CLIPS_DIR
            / f"clip_{index:02d}_thumbnail.jpg"
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"🎬 Processing clip "
            f"{index}/{len(clips)}"
        )

        print(
            f"Video: {rendered_path}"
        )

        print(
            f"Time: "
            f"{float(clip['start']):.2f} → "
            f"{float(clip['end']):.2f}"
        )

        print(
            f"Duration: "
            f"{float(clip['end']) - float(clip['start']):.2f}s"
        )

        print(
            f"Subtitle: "
            f"{subtitle_path}"
        )

        print(
            f"Final: "
            f"{final_path}"
        )

        print(
            f"Metadata: "
            f"{metadata_path}"
        )

        print(
            f"Thumbnail: "
            f"{thumbnail_path}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # SUBTITLES
        # ----------------------------------------------------

        create_ass_subtitles(
            transcript["words"],
            float(
                clip["start"]
            ),
            float(
                clip["end"]
            ),
            subtitle_path,
        )

        burn_subtitles(
            rendered_path,
            subtitle_path,
            final_path,
        )

        if not final_path.exists():

            raise RuntimeError(
                "❌ Final subtitle "
                "video was not created: "
                f"{final_path}"
            )

        final_videos.append(
            str(final_path)
        )

        print()
        print(
            "✅ Subtitle video created"
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        print()

        print(
            f"🤖 Generating metadata "
            f"for clip {index}..."
        )

        metadata = generate_metadata(
            clip
        )

        if not metadata:

            raise RuntimeError(
                "❌ Metadata generation "
                f"failed for clip {index}"
            )

        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        metadata_files.append(
            str(metadata_path)
        )

        print()
        print(
            "✅ Metadata created"
        )

        print(
            f"Title: "
            f"{metadata.get('title', '')}"
        )

        print(
            "Description:"
        )

        print(
            metadata.get(
                "description",
                "",
            )
        )

        print(
            "Hashtags: "
            + " ".join(
                metadata.get(
                    "hashtags",
                    [],
                )
            )
        )

        print(
            "Tags: "
            + ", ".join(
                metadata.get(
                    "tags",
                    [],
                )
            )
        )

        print(
            f"📁 {metadata_path}"
        )

        # ----------------------------------------------------
        # THUMBNAIL
        # ----------------------------------------------------

        print()

        print(
            f"🖼️ Generating thumbnail "
            f"for clip {index}..."
        )

        create_thumbnail(
            final_path,
            metadata_path,
            thumbnail_path,
        )

        if not thumbnail_path.exists():

            raise RuntimeError(
                "❌ Thumbnail was not created: "
                f"{thumbnail_path}"
            )

        thumbnail_files.append(
            str(thumbnail_path)
        )

        print(
            "✅ Thumbnail created"
        )

        print(
            f"📁 {thumbnail_path}"
        )

        # ----------------------------------------------------
        # YOUTUBE UPLOAD
        # ----------------------------------------------------

        print()

        print(
            "=" * 60
        )

        print(
            f"📤 UPLOADING CLIP "
            f"{index}/{len(clips)} TO YOUTUBE"
        )

        print(
            "=" * 60
        )

        title = str(
            metadata.get(
                "title",
                "",
            )
        ).strip()

        description = str(
            metadata.get(
                "description",
                "",
            )
        ).strip()

        hashtags = metadata.get(
            "hashtags",
            [],
        )

        tags = metadata.get(
            "tags",
            [],
        )

        if not title:

            raise RuntimeError(
                f"❌ Empty YouTube title "
                f"for clip {index}"
            )

        if not description:

            raise RuntimeError(
                f"❌ Empty YouTube description "
                f"for clip {index}"
            )

        if not isinstance(
            hashtags,
            list,
        ):

            hashtags = []

        if not isinstance(
            tags,
            list,
        ):

            tags = []

        hashtags_text = " ".join(
            str(item).strip()
            for item in hashtags
            if str(item).strip()
        )

        clean_tags = []

        for tag in tags:

            tag = str(
                tag
            ).strip()

            if not tag:
                continue

            if tag.startswith("#"):
                tag = tag[1:]

            if tag not in clean_tags:
                clean_tags.append(
                    tag
                )

        youtube_video_id = upload_video(
            video_path=str(
                final_path
            ),
            thumbnail_path=str(
                thumbnail_path
            ),
            title=title,
            description=description,
            hashtags=hashtags_text,
            tags=clean_tags,
        )

        if not youtube_video_id:

            raise RuntimeError(
                f"❌ YouTube upload failed "
                f"for clip {index}"
            )

        uploaded_videos.append(
            youtube_video_id
        )

        print()
        print(
            f"✅ Clip {index} published "
            f"to YouTube"
        )

        print(
            f"🎬 Video ID: "
            f"{youtube_video_id}"
        )

        print(
            "🔗 "
            f"https://www.youtube.com/watch?v="
            f"{youtube_video_id}"
        )

        # ====================================================
        # ВАЖНО:
        #
        # Записываем клип в историю СРАЗУ после успешного
        # upload.
        #
        # Поэтому если следующий клип упадёт с ошибкой,
        # первый уже не будет опубликован повторно.
        # ====================================================

        if SELECTED_SOURCE_VIDEO_INFO:

            history_saved = (
                record_published_clip(
                    SELECTED_SOURCE_VIDEO_INFO,
                    clip,
                    youtube_video_id,
                )
            )

            if not history_saved:

                print(
                    "⚠️ WARNING: YouTube upload "
                    "succeeded but history save failed."
                )

            else:

                # Сразу пытаемся отправить history
                # в GitHub, чтобы следующий Actions run
                # уже знал об этом опубликованном клипе.
                persist_history_to_git()

        else:

            print(
                "⚠️ Source metadata unavailable. "
                "Published clip could not be linked "
                "to a RUTUBE source."
            )

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "🎉 SHORTS PIPELINE READY"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # FINAL VIDEOS
    # --------------------------------------------------------

    print(
        "🎬 Final videos:"
    )

    for index, video_path in enumerate(
        final_videos,
        start=1,
    ):

        path = Path(
            video_path
        )

        if not path.exists():
            continue

        size_mb = (
            path.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"  ✅ Clip {index}: "
            f"{path}"
        )

        print(
            f"     Size: "
            f"{size_mb:.2f} MB"
        )

    print()

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    print(
        "📝 Metadata files:"
    )

    for index, metadata_path in enumerate(
        metadata_files,
        start=1,
    ):

        print(
            f"  ✅ Clip {index}: "
            f"{metadata_path}"
        )

    print()

    # --------------------------------------------------------
    # THUMBNAILS
    # --------------------------------------------------------

    print(
        "🖼️ Thumbnail files:"
    )

    for index, thumbnail_path in enumerate(
        thumbnail_files,
        start=1,
    ):

        path = Path(
            thumbnail_path
        )

        if not path.exists():
            continue

        size_mb = (
            path.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"  ✅ Clip {index}: "
            f"{path}"
        )

        print(
            f"     Size: "
            f"{size_mb:.2f} MB"
        )

    print()

    # --------------------------------------------------------
    # YOUTUBE
    # --------------------------------------------------------

    print(
        "📺 YouTube uploads:"
    )

    for index, video_id in enumerate(
        uploaded_videos,
        start=1,
    ):

        print(
            f"  ✅ Clip {index}: "
            f"{video_id}"
        )

        print(
            f"     https://www.youtube.com/watch?v="
            f"{video_id}"
        )

    print()

    # --------------------------------------------------------
    # FINAL COUNTS
    # --------------------------------------------------------

    print(
        "📊 Pipeline summary:"
    )

    print(
        f"   New clips selected: "
        f"{len(clips)}"
    )

    print(
        f"   Videos rendered:    "
        f"{len(final_videos)}"
    )

    print(
        f"   Metadata created:   "
        f"{len(metadata_files)}"
    )

    print(
        f"   Thumbnails created: "
        f"{len(thumbnail_files)}"
    )

    print(
        f"   YouTube uploads:    "
        f"{len(uploaded_videos)}"
    )

    print()

    if len(final_videos) != len(
        clips
    ):

        raise RuntimeError(
            "❌ Final video count does not "
            "match selected clip count"
        )

    if len(metadata_files) != len(
        clips
    ):

        raise RuntimeError(
            "❌ Metadata count does not "
            "match selected clip count"
        )

    if len(thumbnail_files) != len(
        clips
    ):

        raise RuntimeError(
            "❌ Thumbnail count does not "
            "match selected clip count"
        )

    if len(uploaded_videos) != len(
        clips
    ):

        raise RuntimeError(
            "❌ YouTube upload count does not "
            "match selected clip count"
        )

    print(
        "=" * 70
    )

    print(
        "✅ PROCESS COMPLETE"
    )

    print(
        "📺 All NEW selected Shorts uploaded to YouTube"
    )

    print(
        "🔁 Source remains available "
        "for future re-processing."
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "🚀 RUTUBE AI SHORTS"
    )

    print()

    if "--download" in sys.argv:

        print(
            "Mode: DOWNLOAD"
        )

        print()

        find_and_download()

        return

    if "--process" in sys.argv:

        print(
            "Mode: PROCESS"
        )

        print()

        process_video()

        return

    print(
        "Mode: DOWNLOAD + PROCESS"
    )

    print()

    find_and_download()

    print()

    process_video()


if __name__ == "__main__":
    main()
