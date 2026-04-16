"""
Central Gemini client — all four FairLens stages use this.
Calls the Gemini REST API directly via httpx (v1 stable endpoint, not v1beta).
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

# Stable v1 endpoint — avoids the v1beta model availability issues
_BASE_URL = "https://generativelanguage.googleapis.com/v1/models"

# Model fallback chain — tried in order until one succeeds
_MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def _candidate_models(preferred: str) -> list[str]:
    return [preferred] + [m for m in _MODEL_FALLBACK_CHAIN if m != preferred]


async def ask_gemini(prompt: str, model: str = "pro", expect_json: bool = False) -> str:
    """
    Send a prompt to Gemini and return the text response.
    Uses the v1 stable REST API directly.
    Automatically falls back through the model chain on 403/404 errors.
    """
    preferred = settings.GEMINI_MODEL if model == "pro" else settings.GEMINI_FLASH_MODEL
    last_err = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        for model_name in _candidate_models(preferred):
            url = f"{_BASE_URL}/{model_name}:generateContent"
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

                if resp.status_code in (403, 404):
                    logger.warning(
                        f"Gemini model '{model_name}' unavailable "
                        f"(HTTP {resp.status_code}), trying next..."
                    )
                    last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    continue

                resp.raise_for_status()
                data = resp.json()

                if model_name != preferred:
                    logger.info(f"Gemini: fell back to '{model_name}' (primary '{preferred}' unavailable)")

                text = data["candidates"][0]["content"]["parts"][0]["text"]

                if expect_json:
                    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
                    text = re.sub(r"\n?```$", "", text.strip())

                return text

            except httpx.HTTPStatusError as e:
                logger.error(f"Gemini HTTP error with model '{model_name}': {e}")
                raise
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected Gemini response structure: {data}")
                raise ValueError(f"Could not parse Gemini response: {e}") from e

    logger.error(f"All Gemini models exhausted. Last error: {last_err}")
    raise last_err or RuntimeError("All Gemini models failed")


async def ask_gemini_json(prompt: str, model: str = "flash") -> dict:
    """Ask Gemini and parse the response as JSON."""
    text = await ask_gemini(prompt, model=model, expect_json=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Gemini did not return valid JSON. Got: {text[:300]}")
