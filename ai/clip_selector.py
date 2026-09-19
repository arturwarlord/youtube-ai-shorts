import os
import json
import re
from typing import List, Dict, Any

from google import genai

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


MODEL = "gemini-flash-lite-latest"

MIN_CLIP_DURATION = 20
MAX_CLIP_DURATION = 60


def _get_client():
    api_key = os.getenv("GEMINI_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_KEY is not set")

    return genai.Client(api_key=api_key)


def _clean_response_text(text: str) -> str:
    """
    Remove common Markdown wrappers around Gemini JSON.
    """
    text = (text or "").strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _candidate_json_strings(text: str):
    """
    Return likely JSON portions of a Gemini response.
    """
    text = _clean_response_text(text)

    candidates = [text]

    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    return candidates


def _repair_json_fallback(text: str):
    """
    Small fallback for common Gemini formatting mistakes.
    """
    repaired = text.strip()

    repaired = re.sub(
        r",\s*([}\]])",
        r"\1",
        repaired,
    )

    repaired = (
        repaired
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    return repaired


def _extract_json(text: str):
    """
    Extract and parse JSON from Gemini response.
    """
    if not text or not text.strip():
        raise ValueError("Gemini returned empty JSON response")

    candidates = _candidate_json_strings(text)

    # 1. Strict JSON.
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2. json-repair.
    if repair_json is not None:
        for candidate in candidates:
            try:
                repaired = repair_json(candidate)
                parsed = json.loads(repaired)

                return parsed

            except Exception:
                pass

    # 3. Lightweight repair.
    for candidate in candidates:
        try:
            repaired = _repair_json_fallback(candidate)

            return json.loads(repaired)

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Gemini returned invalid JSON after all parsing/repair attempts"
    )


def _format_transcript(words: List[Dict]) -> str:
    """
    Convert word-level Whisper timestamps into compact transcript.
    """
    lines = []

    for word in words:
        text = str(word.get("word", "")).strip()

        if not text:
            continue

        try:
            start = float(word.get("start", 0))
        except (TypeError, ValueError):
            start = 0.0

        try:
            end = float(word.get("end", start))
        except (TypeError, ValueError):
            end = start

        lines.append(
            f"[{start:.2f}-{end:.2f}] {text}"
        )

    return "\n".join(lines)


def _parse_timestamp(value: Any):
    """
    Convert different Gemini timestamp formats to seconds.

    Supported:
        123.45
        "123.45"
        "01:23"
        "00:01:23"
        "1m 23s"
        "83 sec"
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if not text:
        return None

    # Plain number.
    try:
        return float(text.replace(",", "."))
    except ValueError:
        pass

    # HH:MM:SS / MM:SS
    if ":" in text:
        parts = text.split(":")

        try:
            parts = [float(p.replace(",", ".")) for p in parts]

            if len(parts) == 3:
                hours, minutes, seconds = parts

                return (
                    hours * 3600
                    + minutes * 60
                    + seconds
                )

            if len(parts) == 2:
                minutes, seconds = parts

                return minutes * 60 + seconds

        except ValueError:
            pass

    # 1h 23m 45s / 23m 45s / 45s
    match = re.fullmatch(
        r"\s*"
        r"(?:(\d+(?:\.\d+)?)\s*h)?"
        r"\s*"
        r"(?:(\d+(?:\.\d+)?)\s*m)?"
        r"\s*"
        r"(?:(\d+(?:\.\d+)?)\s*s)?"
        r"\s*",
        text.lower(),
    )

    if match:
        hours = float(match.group(1) or 0)
        minutes = float(match.group(2) or 0)
        seconds = float(match.group(3) or 0)

        if hours or minutes or seconds:
            return (
                hours * 3600
                + minutes * 60
                + seconds
            )

    # Extract first numeric value as last-resort fallback.
    match = re.search(
        r"\d+(?:[.,]\d+)?",
        text,
    )

    if match:
        try:
            return float(
                match.group(0).replace(",", ".")
            )
        except ValueError:
            pass

    return None


def _normalize_candidate(candidate: Dict) -> Dict:
    """
    Normalize a Gemini candidate before validation.

    This does NOT change Gemini's selection logic.
    It only makes the returned structure safer to process.
    """
    normalized = dict(candidate)

    start = _parse_timestamp(
        normalized.get("start")
    )

    end = _parse_timestamp(
        normalized.get("end")
    )

    if start is not None:
        normalized["start"] = round(start, 2)

    if end is not None:
        normalized["end"] = round(end, 2)

    # Normalize score if present.
    try:
        normalized["score"] = float(
            normalized.get("score", 0)
        )
    except (TypeError, ValueError):
        normalized["score"] = 0

    # Ensure textual fields exist.
    normalized["reason"] = str(
        normalized.get("reason", "")
    ).strip()

    normalized["hook"] = str(
        normalized.get("hook", "")
    ).strip()

    normalized["title_hint"] = str(
        normalized.get("title_hint", "")
    ).strip()

    return normalized


def _validate_candidate(candidate: Dict) -> bool:
    """
    Validate a Gemini candidate.

    The validation is intentionally tolerant:
    - timestamps may be numbers or strings;
    - small timestamp-format differences are normalized;
    - duration must still remain within Shorts limits.
    """
    if not isinstance(candidate, dict):
        return False

    required = [
        "start",
        "end",
        "reason",
        "hook",
    ]

    if not all(key in candidate for key in required):
        return False

    start = _parse_timestamp(candidate.get("start"))
    end = _parse_timestamp(candidate.get("end"))

    if start is None or end is None:
        return False

    # Invalid timestamps.
    if start < 0:
        return False

    if end <= start:
        return False

    duration = end - start

    # Keep the original required Shorts duration.
    if duration < MIN_CLIP_DURATION:
        return False

    if duration > MAX_CLIP_DURATION:
        return False

    candidate["start"] = round(start, 2)
    candidate["end"] = round(end, 2)

    # Normalize score.
    try:
        candidate["score"] = float(
            candidate.get("score", 0)
        )
    except (TypeError, ValueError):
        candidate["score"] = 0

    # Normalize text fields.
    candidate["reason"] = str(
        candidate.get("reason", "")
    ).strip()

    candidate["hook"] = str(
        candidate.get("hook", "")
    ).strip()

    candidate["title_hint"] = str(
        candidate.get("title_hint", "")
    ).strip()

    return True


def _remove_overlaps(clips: List[Dict]) -> List[Dict]:
    """
    Remove heavily overlapping clips.

    Higher scored clips are kept.
    """
    clips = sorted(
        clips,
        key=lambda x: float(x.get("score", 0)),
        reverse=True,
    )

    result = []

    for clip in clips:
        start = float(clip["start"])
        end = float(clip["end"])

        overlaps = False

        for existing in result:
            existing_start = float(existing["start"])
            existing_end = float(existing["end"])

            intersection = max(
                0,
                min(end, existing_end)
                - max(start, existing_start),
            )

            current_duration = end - start
            existing_duration = (
                existing_end - existing_start
            )

            smaller_duration = min(
                current_duration,
                existing_duration,
            )

            if smaller_duration <= 0:
                continue

            overlap_ratio = (
                intersection / smaller_duration
            )

            if overlap_ratio >= 0.5:
                overlaps = True
                break

        if not overlaps:
            result.append(clip)

    return sorted(
        result,
        key=lambda x: x["start"],
    )


def select_clips(
    words: List[Dict],
    max_clips: int = 3,
) -> List[Dict]:
    """
    Select the strongest Shorts-worthy moments
    from a long-video transcript.

    Returns:
        [
            {
                "start": 123.4,
                "end": 165.8,
                "score": 91,
                "hook": "...",
                "reason": "...",
                "title_hint": "..."
            }
        ]
    """

    if not words:
        raise ValueError("Transcript is empty")

    transcript = _format_transcript(words)

    client = _get_client()

    prompt = f"""
You are an expert YouTube Shorts editor.

Analyze the transcript of a long-form video below and identify the
BEST self-contained moments that can become viral Shorts.

IMPORTANT:
- Select moments between {MIN_CLIP_DURATION} and {MAX_CLIP_DURATION} seconds.
- A clip must make sense WITHOUT the viewer watching the full video.
- Prefer a strong hook in the first few seconds.
- Prefer a clear story, explanation, surprising fact, argument,
  emotional moment, useful insight, or strong payoff.
- The clip should have a natural beginning and ending.
- Avoid greetings.
- Avoid introductions like "today we are going to..."
- Avoid advertisements.
- Avoid sponsor segments.
- Avoid long pauses.
- Avoid moments that require previous context.
- Avoid sentences that begin with "as I said earlier" or similar.
- Do not invent words or facts.
- Use only timestamps present in the transcript.
- Do not create overlapping clips when possible.
- Find up to 10 strong candidates.

Score every candidate from 0 to 100 using:

HOOK:
Does it immediately create curiosity?

STORY:
Does the moment develop naturally?

PAYOFF:
Does it deliver a satisfying conclusion or insight?

STANDALONE:
Can someone understand it without the rest of the video?

EMOTIONAL:
Does it create curiosity, surprise, tension, admiration,
fear, amusement or another strong reaction?

CLARITY:
Is the idea easy to understand?

Return ONLY valid JSON.

Format:
[
  {{
    "start": 123.40,
    "end": 165.80,
    "score": 91,
    "hook": "Short description of why the opening is strong",
    "reason": "Why this entire segment works as a Short",
    "title_hint": "Potential title idea",
    "hook_score": 9,
    "story_score": 9,
    "payoff_score": 9,
    "standalone_score": 10,
    "emotional_score": 8,
    "clarity_score": 9
  }}
]

TRANSCRIPT:

{transcript}
"""

    print("🤖 Calling Gemini clip selector...")

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini returned an empty response"
        )

    print("✅ Gemini response received")

    candidates = _extract_json(response.text)

    if not isinstance(candidates, list):
        raise ValueError(
            "Gemini response must be a JSON array"
        )

    print(
        f"📊 Gemini returned {len(candidates)} candidates"
    )

    valid = []

    for index, candidate in enumerate(candidates, start=1):

        if not isinstance(candidate, dict):
            print(
                f"⚠️ Candidate #{index}: "
                f"not an object, skipped"
            )
            continue

        normalized = _normalize_candidate(
            candidate
        )

        if _validate_candidate(normalized):
            valid.append(normalized)

            print(
                f"✅ Candidate #{index}: "
                f"{normalized['start']:.2f}s → "
                f"{normalized['end']:.2f}s "
                f"({normalized['end'] - normalized['start']:.2f}s) "
                f"score={normalized.get('score', 0)}"
            )

        else:
            start = normalized.get("start")
            end = normalized.get("end")

            print(
                f"⚠️ Candidate #{index}: invalid "
                f"start={start}, end={end}"
            )

    if not valid:
        print(
            "❌ Gemini returned candidates, "
            "but none passed validation."
        )

        # Print a compact diagnostic instead of silently failing.
        for index, candidate in enumerate(
            candidates[:10],
            start=1,
        ):
            if isinstance(candidate, dict):
                print(
                    f"   Candidate #{index}: "
                    f"start={candidate.get('start')!r}, "
                    f"end={candidate.get('end')!r}, "
                    f"score={candidate.get('score')!r}"
                )

        raise RuntimeError(
            "No valid Shorts candidates found"
        )

    valid = _remove_overlaps(valid)

    valid.sort(
        key=lambda x: float(
            x.get("score", 0)
        ),
        reverse=True,
    )

    selected = valid[:max_clips]

    print(
        f"🎯 Selected {len(selected)} Shorts candidates"
    )

    for index, clip in enumerate(
        selected,
        start=1,
    ):
        print(
            f"   #{index}: "
            f"{clip['start']:.2f}s → "
            f"{clip['end']:.2f}s | "
            f"score={clip.get('score', 0)}"
        )

    return selected
