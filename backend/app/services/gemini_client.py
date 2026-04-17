"""
Gemini client using the official google-generativeai SDK.
Set GEMINI_API_KEY in .env — free key at https://aistudio.google.com/app/apikey
"""
import logging
import json
import re

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

from app.core.config import settings

logger = logging.getLogger(__name__)

_MODEL = "gemini-2.5-flash"

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set — AI stages will fail.")

_generation_config = genai.GenerationConfig(temperature=0.2, max_output_tokens=8192)


async def ask_gemini(prompt: str, expect_json: bool = False) -> str:
    """Send a prompt to Gemini 2.5 Flash and return the text response."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    try:
        model = genai.GenerativeModel(_MODEL, generation_config=_generation_config)
        response = await model.generate_content_async(prompt)
        text = response.text
    except GoogleAPIError as e:
        logger.error(f"Gemini API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise

    if expect_json:
        text = re.sub(r"^```(?:json)?\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text.strip())

    return text


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
