"""
Central Gemini client — all four FairLens stages use this.
Calls the Gemini REST API directly via httpx.

On any non-200 response the client continues to the next candidate rather than
raising immediately. Only network errors (DNS, timeout) abort early.

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

# Tried in order until one returns HTTP 200.
# v1beta generic aliases work with all API key tiers (including the free tier).
# Versioned v1 names are included as a last-resort fallback.
_CANDIDATES = [
    ("v1beta", "gemini-2.5-flash"),
    ("v1beta", "gemini-2.5-flash-preview-04-17"),
    ("v1beta", "gemini-2.0-flash"),
    ("v1",     "gemini-2.0-flash-001"),
    ("v1beta", "gemini-1.5-flash"),
    ("v1",     "gemini-1.5-flash-001"),
]


async def ask_gemini(prompt: str, expect_json: bool = False) -> str:
    """
    Send a prompt to Gemini and return the text response.

    Tries every candidate in _CANDIDATES.  Any non-200 HTTP response is
    logged and skipped; only a genuine network error (timeout, DNS) raises
    before the list is exhausted.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )

    last_err: Exception | None = None

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
            except httpx.RequestError as e:
                # Network-level error (timeout, DNS failure, etc.) — try next
                logger.warning(f"Gemini network error ({api_version}/{model_name}): {e}")
                last_err = e
                continue

            if resp.status_code != 200:
                # Any HTTP error: model not found, quota exceeded, region blocked, etc.
                logger.warning(
                    f"Gemini {api_version}/{model_name}: HTTP {resp.status_code} — trying next. "
                    f"Body: {resp.text[:120]}"
                )
                last_err = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            # HTTP 200 — parse the response
            try:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Gemini {api_version}/{model_name}: unexpected response shape — trying next. "
                    f"Error: {e}. Body: {resp.text[:120]}"
                )
                last_err = ValueError(f"Unexpected Gemini response shape: {e}")
                continue

            logger.info(f"Gemini: success with {api_version}/{model_name}")

            if expect_json:
                text = re.sub(r"^```(?:json)?\n?", "", text.strip())
                text = re.sub(r"\n?```$", "", text.strip())

            return text

    # All candidates exhausted
    logger.error(f"All Gemini candidates failed. Last error: {last_err}")
    raise last_err or RuntimeError("All Gemini models failed — check your GEMINI_API_KEY")


async def ask_gemini_json(prompt: str) -> dict:
    """Ask Gemini and parse the response as JSON."""
    text = await ask_gemini(prompt, expect_json=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Gemini did not return valid JSON. Got: {text[:300]}")
