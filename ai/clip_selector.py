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
            parts = [
                float(p.replace(",", "."))
                for p in parts
            ]

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


def _get_transcript_duration(words: List[Dict]) -> float:
    """
    Get approximate source duration from Whisper timestamps.

    Uses the latest valid word end timestamp.
    """
    latest_end = 0.0

    for word in words:
        try:
            end = float(word.get("end", 0))
        except (TypeError, ValueError):
            continue

        if end > latest_end:
            latest_end = end

    return latest_end


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


def _expand_short_candidate(
    candidate: Dict,
    source_duration: float,
) -> bool:
    """
    Automatically expand a candidate shorter than MIN_CLIP_DURATION.

    Gemini sometimes returns a very good moment that is shorter than
    the requested minimum. Instead of throwing it away, expand the
    surrounding context until the clip reaches MIN_CLIP_DURATION.

    Strategy:
        1. Prefer extending the END.
        2. If there is not enough room at the end, extend START.
        3. Never go outside the source video.
        4. Never exceed MAX_CLIP_DURATION.
    """
    start = _parse_timestamp(candidate.get("start"))
    end = _parse_timestamp(candidate.get("end"))

    if start is None or end is None:
        return False

    if start < 0:
        start = 0.0

    if end <= start:
        return False

    # If we don't know source duration, don't attempt expansion.
    if source_duration <= 0:
        return False

    # Clamp to actual source duration.
    start = min(start, source_duration)
    end = min(end, source_duration)

    duration = end - start

    # Already valid.
    if duration >= MIN_CLIP_DURATION:
        candidate["start"] = round(start, 2)
        candidate["end"] = round(end, 2)
        return True

    needed = MIN_CLIP_DURATION - duration

    # ---------------------------------------------------------
    # First: extend the end.
    # ---------------------------------------------------------
    available_after = max(
        0.0,
        source_duration - end,
    )

    extend_end = min(
        needed,
        available_after,
    )

    end += extend_end
    needed -= extend_end

    # ---------------------------------------------------------
    # Second: extend the start if necessary.
    # ---------------------------------------------------------
    if needed > 0:
        available_before = max(
            0.0,
            start,
        )

        extend_start = min(
            needed,
            available_before,
        )

        start -= extend_start
        needed -= extend_start

    duration = end - start

    # Still too short means there isn't enough source material.
    if duration < MIN_CLIP_DURATION:
        return False

    # Safety cap.
    if duration > MAX_CLIP_DURATION:
        end = start + MAX_CLIP_DURATION

        if end > source_duration:
            end = source_duration
            start = max(
                0.0,
                end - MAX_CLIP_DURATION,
            )

    candidate["start"] = round(start, 2)
    candidate["end"] = round(end, 2)

    return True


def _validate_candidate(
    candidate: Dict,
    source_duration: float = 0.0,
) -> bool:
    """
    Validate a Gemini candidate.

    Validation is tolerant:
    - timestamps may be numbers or strings;
    - timestamp formats are normalized;
    - short Gemini candidates are automatically expanded;
    - duration remains within Shorts limits;
    - timestamps cannot exceed source duration.
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

    start = _parse_timestamp(
        candidate.get("start")
    )

    end = _parse_timestamp(
        candidate.get("end")
    )

    if start is None or end is None:
        return False

    # Invalid timestamps.
    if start < 0:
        return False

    if end <= start:
        return False

    # If source duration is known, make sure Gemini did not
    # return a timestamp outside the actual video.
    if source_duration > 0:

        if start >= source_duration:
            return False

        if end > source_duration:
            end = source_duration

        if end <= start:
            return False

        candidate["start"] = start
        candidate["end"] = end

    duration = end - start

    # ---------------------------------------------------------
    # Important fix:
    #
    # Gemini sometimes ignores the 20-second minimum and returns
    # 10–19 second moments.
    #
    # Instead of rejecting those potentially strong moments,
    # automatically expand them to at least 20 seconds.
    # ---------------------------------------------------------
    if duration < MIN_CLIP_DURATION:

        expanded = _expand_short_candidate(
            candidate,
            source_duration,
        )

        if not expanded:
            return False

        start = float(candidate["start"])
        end = float(candidate["end"])

        duration = end - start

    # Duration must now be valid.
    if duration < MIN_CLIP_DURATION:
        return False

    if duration > MAX_CLIP_DURATION:

        # Trim from the end first.
        end = start + MAX_CLIP_DURATION

        if source_duration > 0 and end > source_duration:
            end = source_duration
            start = max(
                0.0,
                end - MAX_CLIP_DURATION,
            )

        candidate["start"] = round(start, 2)
        candidate["end"] = round(end, 2)

        duration = end - start

    if duration < MIN_CLIP_DURATION:
        return False

    candidate["start"] = round(
        float(candidate["start"]),
        2,
    )

    candidate["end"] = round(
        float(candidate["end"]),
        2,
    )

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


def _remove_overlaps(
    clips: List[Dict],
) -> List[Dict]:
    """
    Remove heavily overlapping clips.

    Higher scored clips are kept.
    """
    clips = sorted(
        clips,
        key=lambda x: float(
            x.get("score", 0)
        ),
        reverse=True,
    )

    result = []

    for clip in clips:

        start = float(clip["start"])
        end = float(clip["end"])

        overlaps = False

        for existing in result:

            existing_start = float(
                existing["start"]
            )

            existing_end = float(
                existing["end"]
            )

            intersection = max(
                0,
                min(end, existing_end)
                - max(start, existing_start),
            )

            current_duration = end - start

            existing_duration = (
                existing_end
                - existing_start
            )

            smaller_duration = min(
                current_duration,
                existing_duration,
            )

            if smaller_duration <= 0:
                continue

            overlap_ratio = (
                intersection
                / smaller_duration
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
        raise ValueError(
            "Transcript is empty"
        )

    transcript = _format_transcript(words)

    # Approximate source duration from Whisper.
    source_duration = _get_transcript_duration(
        words
    )

    if source_duration > 0:
        print(
            f"⏱ Transcript duration: "
            f"{source_duration:.2f}s"
        )

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

    print(
        "🤖 Calling Gemini clip selector..."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini returned an empty response"
        )

    print(
        "✅ Gemini response received"
    )

    candidates = _extract_json(
        response.text
    )

    if not isinstance(candidates, list):
        raise ValueError(
            "Gemini response must be a JSON array"
        )

    print(
        f"📊 Gemini returned "
        f"{len(candidates)} candidates"
    )

    valid = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        if not isinstance(candidate, dict):
            print(
                f"⚠️ Candidate #{index}: "
                f"not an object, skipped"
            )
            continue

        normalized = _normalize_candidate(
            candidate
        )

        original_start = normalized.get(
            "start"
        )

        original_end = normalized.get(
            "end"
        )

        if _validate_candidate(
            normalized,
            source_duration,
        ):

            valid.append(normalized)

            final_start = normalized["start"]
            final_end = normalized["end"]

            final_duration = (
                final_end
                - final_start
            )

            if (
                original_start is not None
                and original_end is not None
                and (
                    abs(
                        final_start
                        - float(original_start)
                    ) > 0.01
                    or abs(
                        final_end
                        - float(original_end)
                    ) > 0.01
                )
            ):
                print(
                    f"🔧 Candidate #{index}: "
                    f"expanded "
                    f"{float(original_end) - float(original_start):.2f}s "
                    f"→ {final_duration:.2f}s"
                )

            print(
                f"✅ Candidate #{index}: "
                f"{final_start:.2f}s → "
                f"{final_end:.2f}s "
                f"({final_duration:.2f}s) "
                f"score="
                f"{normalized.get('score', 0)}"
            )

        else:

            start = normalized.get(
                "start"
            )

            end = normalized.get(
                "end"
            )

            print(
                f"⚠️ Candidate #{index}: "
                f"invalid "
                f"start={start}, "
                f"end={end}"
            )

    if not valid:

        print(
            "❌ Gemini returned candidates, "
            "but none passed validation."
        )

        # Print compact diagnostic.
        for index, candidate in enumerate(
            candidates[:10],
            start=1,
        ):

            if isinstance(
                candidate,
                dict,
            ):

                start = _parse_timestamp(
                    candidate.get("start")
                )

                end = _parse_timestamp(
                    candidate.get("end")
                )

                duration = None

                if (
                    start is not None
                    and end is not None
                ):
                    duration = end - start

                print(
                    f"   Candidate #{index}: "
                    f"start="
                    f"{candidate.get('start')!r}, "
                    f"end="
                    f"{candidate.get('end')!r}, "
                    f"duration="
                    f"{duration!r}, "
                    f"score="
                    f"{candidate.get('score')!r}"
                )

        raise RuntimeError(
            "No valid Shorts candidates found"
        )

    valid = _remove_overlaps(
        valid
    )

    valid.sort(
        key=lambda x: float(
            x.get("score", 0)
        ),
        reverse=True,
    )

    selected = valid[:max_clips]

    print(
        f"🎯 Selected "
        f"{len(selected)} Shorts candidates"
    )

    for index, clip in enumerate(
        selected,
        start=1,
    ):

        print(
            f"   #{index}: "
            f"{clip['start']:.2f}s → "
            f"{clip['end']:.2f}s | "
            f"duration="
            f"{clip['end'] - clip['start']:.2f}s | "
            f"score="
            f"{clip.get('score', 0)}"
        )

    return selected
