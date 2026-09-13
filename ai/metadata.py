import os
import json
import re

from google import genai


MODEL = "gemini-flash-lite-latest"


def _get_client():
    api_key = os.getenv("GEMINI_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_KEY is not set")

    return genai.Client(api_key=api_key)


def _extract_json(text: str):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"```$",
            "",
            text,
        ).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        match = re.search(
            r"\{[\s\S]*\}",
            text,
        )

        if not match:
            raise ValueError(
                "Gemini returned invalid JSON"
            )

        return json.loads(
            match.group(0)
        )


def generate_metadata(clip):
    """
    Generate YouTube Shorts metadata.

    Returns:
        {
            "title": "...",
            "description": "...",
            "hashtags": [
                "#Shorts",
                "#..."
            ],
            "tags": [
                "...",
                "..."
            ]
        }
    """

    client = _get_client()

    title_hint = clip.get(
        "title_hint",
        "",
    )

    hook = clip.get(
        "hook",
        "",
    )

    reason = clip.get(
        "reason",
        "",
    )

    prompt = f"""
You are an expert YouTube Shorts SEO editor.

Create metadata for a Russian-language YouTube Short.

The Short comes from a podcast/science discussion.

IMPORTANT:
- Everything must be in Russian.
- Do not invent facts.
- Do not use clickbait that contradicts the clip.
- Make the title highly clickable but accurate.
- The title must be short and natural.
- Maximum title length: 90 characters.
- Prefer 45-75 characters.
- Do not put #Shorts in the title.
- Description should be 2-4 short sentences.
- Description should explain the topic and create curiosity.
- Do not mention AI.
- Do not mention that the video was generated automatically.
- Do not include links.
- Do not include timestamps.
- Generate 5-8 relevant hashtags.
- Always include #Shorts.
- Generate 8-15 YouTube search tags.
- Tags should be normal search phrases, without #.
- Avoid generic spam tags.
- Do not use unrelated trending topics.

CLIP INFORMATION:

Title hint:
{title_hint}

Hook:
{hook}

Reason:
{reason}

Return ONLY valid JSON.

Format:

{{
  "title": "Название Shorts",
  "description": "Описание в 2-4 предложениях.",
  "hashtags": [
    "#Shorts",
    "#Наука",
    "#..."
  ],
  "tags": [
    "наука",
    "физика",
    "..."
  ]
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini returned empty metadata"
        )

    data = _extract_json(
        response.text
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Metadata must be a JSON object"
        )

    title = str(
        data.get("title", "")
    ).strip()

    description = str(
        data.get("description", "")
    ).strip()

    hashtags = data.get(
        "hashtags",
        [],
    )

    tags = data.get(
        "tags",
        [],
    )

    if not title:
        raise RuntimeError(
            "Gemini returned empty title"
        )

    if not description:
        raise RuntimeError(
            "Gemini returned empty description"
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

    # --------------------------------------------------------
    # Нормализуем hashtags
    # --------------------------------------------------------

    clean_hashtags = []

    for hashtag in hashtags:

        hashtag = str(
            hashtag
        ).strip()

        if not hashtag:
            continue

        if not hashtag.startswith("#"):
            hashtag = "#" + hashtag

        if hashtag not in clean_hashtags:
            clean_hashtags.append(
                hashtag
            )

    # #Shorts обязателен
    if "#Shorts" not in clean_hashtags:
        clean_hashtags.insert(
            0,
            "#Shorts",
        )

    # --------------------------------------------------------
    # Нормализуем tags
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Ограничиваем длину title
    # --------------------------------------------------------

    if len(title) > 90:
        title = title[:87].rstrip() + "..."

    return {
        "title": title,
        "description": description,
        "hashtags": clean_hashtags[:8],
        "tags": clean_tags[:15],
    }
