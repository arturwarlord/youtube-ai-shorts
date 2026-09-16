import os
import json
import re
from typing import List, Dict

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

    # Remove ```json ... ``` or ``` ... ```
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

    # Prefer the JSON array because select_clips expects an array.
    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    return candidates


def _repair_json_fallback(text: str):
    """
    Small fallback for common Gemini formatting mistakes when
    json-repair is not installed.

    This does NOT change the clip-selection logic. It only attempts
    to turn an almost-valid JSON response into valid JSON.
    """
    repaired = text.strip()

    # Remove trailing commas before } or ].
    repaired = re.sub(
        r",\s*([}\]])",
        r"\1",
        repaired,
    )

    # Normalize smart quotes that Gemini can occasionally emit.
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

    Gemini normally returns valid JSON, but occasionally returns
    almost-valid JSON (for example a trailing comma or a Markdown
    wrapper). We first use the standard JSON parser, then fall back
    to json-repair if available.

    This function only repairs/parses the response. The selection
    prompt and selection logic remain unchanged.
    """
    if not text or not text.strip():
        raise ValueError("Gemini returned empty JSON response")

    candidates = _candidate_json_strings(text)

    # 1. Strict JSON parsing first.
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2. Use json-repair for malformed JSON from Gemini.
    if repair_json is not None:
        for candidate in candidates:
            try:
                repaired = repair_json(candidate)
                parsed = json.loads(repaired)
                return parsed
            except Exception:
                pass

    # 3. Lightweight fallback if json-repair is unavailable.
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

        start = float(word.get("start", 0))
        end = float(word.get("end", start))

        lines.append(
            f"[{start:.2f}-{end:.2f}] {text}"
        )

    return "\n".join(lines)


def _validate_candidate(candidate: Dict) -> bool:
    required = [
        "start",
        "end",
        "reason",
        "hook",
    ]

    if not all(key in candidate for key in required):
        return False

    try:
        start = float(candidate["start"])
        end = float(candidate["end"])
    except (TypeError, ValueError):
        return False

    duration = end - start

    if duration < MIN_CLIP_DURATION:
        return False

    if duration > MAX_CLIP_DURATION:
        return False

    candidate["start"] = round(start, 2)
    candidate["end"] = round(end, 2)

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
                min(end, existing_end) - max(start, existing_start),
            )

            current_duration = end - start
            existing_duration = existing_end - existing_start

            smaller_duration = min(
                current_duration,
                existing_duration,
            )

            if smaller_duration <= 0:
                continue

            overlap_ratio = intersection / smaller_duration

            if overlap_ratio >= 0.5:
                overlaps = True
                break

        if not overlaps:
            result.append(clip)

    return sorted(
        result,
        key=lambda x: x["start"],
    )


def select_clips(words: List[Dict], max_clips: int = 3) -> List[Dict]:
    """
    Select the strongest Shorts-worthy moments from a long-video transcript.

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

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    if not response or not response.text:
        raise RuntimeError("Gemini returned an empty response")

    candidates = _extract_json(response.text)

    if not isinstance(candidates, list):
        raise ValueError("Gemini response must be a JSON array")

    valid = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        if _validate_candidate(candidate):
            valid.append(candidate)

    if not valid:
        raise RuntimeError("No valid Shorts candidates found")

    valid = _remove_overlaps(valid)

    valid.sort(
        key=lambda x: float(x.get("score", 0)),
        reverse=True,
    )

    return valid[:max_clips]
