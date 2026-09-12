import os
import json
import re
from typing import List, Dict

from google import genai


MODEL = "gemini-flash-lite-latest"
MIN_CLIP_DURATION = 20
MAX_CLIP_DURATION = 60


def _get_client():
    api_key = os.getenv("GEMINI_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_KEY is not set")

    return genai.Client(api_key=api_key)


def _extract_json(text: str):
    """
    Extract JSON from Gemini response.
    Handles both plain JSON and ```json ... ``` responses.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)

        if not match:
            raise ValueError("Gemini returned invalid JSON")

        return json.loads(match.group(0))


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

    return sorted(result, key=lambda x: x["start"])


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
