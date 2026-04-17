"""
Central Gemini client — all four FairLens stages use this.
Calls the Gemini REST API directly via httpx.
Tries v1beta generic names first (broadest API key compatibility), then v1 versioned names.
Set GEMINI_API_KEY in .env — get one free at https://aistudio.google.com/app/apikey
"""
import httpx
import logging
import json
import re
from app.core.config import settings

logger = logging.getLogger(__name__)

if not settings.GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY not set — AI stages will fail. "
        "Get a free key at https://aistudio.google.com/app/apikey"
    )

_BASE = "https://generativelanguage.googleapis.com"

# Each entry is (api_version, model_name).
# v1beta generic aliases work with all API key tiers; v1 versioned names require GA access.
# Tried in order until one returns 200.
_CANDIDATES = [
    ("v1beta", "gemini-2.5-flash"),
    ("v1beta", "gemini-2.0-flash"),
    ("v1beta", "gemini-1.5-flash"),
    ("v1",     "gemini-2.0-flash-001"),
    ("v1",     "gemini-1.5-flash-001"),
    ("v1",     "gemini-1.5-flash-002"),
]


async def ask_gemini(prompt: str, expect_json: bool = False) -> str:
    """
    Send a prompt to Gemini and return the text response.
    Falls back through the candidate list until one model returns 200.
    """
    last_err = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        for api_version, model_name in _CANDIDATES:
            url = f"{_BASE}/{api_version}/models/{model_name}:generateContent"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 8192,
                },
            }
            try:
                resp = await client.post(
                    url,
                    json=payload,
                    params={"key": settings.GEMINI_API_KEY},
                    headers={"Content-Type": "application/json"},
                )

                if resp.status_code in (400, 403, 404):
                    logger.warning(
                        f"Gemini {api_version}/{model_name}: HTTP {resp.status_code}, trying next..."
                    )
                    last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    continue

                resp.raise_for_status()
                data = resp.json()

                logger.debug(f"Gemini: using {api_version}/{model_name}")
                text = data["candidates"][0]["content"]["parts"][0]["text"]

                if expect_json:
                    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
                    text = re.sub(r"\n?```$", "", text.strip())

                return text

            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected Gemini response from {api_version}/{model_name}: {data}")
                raise ValueError(f"Could not parse Gemini response: {e}") from e
            except httpx.HTTPStatusError as e:
                logger.error(f"Gemini HTTP error ({api_version}/{model_name}): {e}")
                raise

    # All candidates exhausted
    key_hint = "GEMINI_API_KEY is not set" if not settings.GEMINI_API_KEY else "check your API key at https://aistudio.google.com/app/apikey"
    logger.error(f"All Gemini models exhausted. {key_hint}. Last error: {last_err}")
    raise last_err or RuntimeError(f"All Gemini models failed — {key_hint}")


async def ask_gemini_json(prompt: str) -> dict:
    """Ask Gemini and parse the response as JSON."""
    text = await ask_gemini(prompt, expect_json=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Gemini did not return valid JSON. Got: {text[:300]}")
