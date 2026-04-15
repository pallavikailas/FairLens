"""
Central Gemini client — all four FairLens stages use this.
Uses google-generativeai SDK directly (no Vertex AI SDK needed locally).
Set GEMINI_API_KEY in .env — get one free at https://aistudio.google.com/app/apikey
"""
import google.generativeai as genai
from app.core.config import settings
import logging, json, re

logger = logging.getLogger(__name__)

# Configure once at import time
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set — AI stages will fail. Get a free key at https://aistudio.google.com/app/apikey")

# Model fallback chain — tried in order until one succeeds
_MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

_model_cache: dict = {}


def _get_model(name: str):
    if name not in _model_cache:
        _model_cache[name] = genai.GenerativeModel(name)
    return _model_cache[name]


def _candidate_models(preferred: str) -> list[str]:
    """Return preferred model first, then the rest of the fallback chain."""
    chain = [preferred] + [m for m in _MODEL_FALLBACK_CHAIN if m != preferred]
    return chain


async def ask_gemini(prompt: str, model: str = "pro", expect_json: bool = False) -> str:
    """
    Send a prompt to Gemini and return the text response.
    Automatically falls back through the model chain on 403/404 errors.
    """
    preferred = settings.GEMINI_MODEL if model == "pro" else settings.GEMINI_FLASH_MODEL
    last_err = None

    for model_name in _candidate_models(preferred):
        try:
            m = _get_model(model_name)
            response = m.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                )
            )
            if model_name != preferred:
                logger.info(f"Gemini: fell back to '{model_name}' (primary '{preferred}' unavailable)")
            text = response.text
            if expect_json:
                text = re.sub(r"^```(?:json)?\n?", "", text.strip())
                text = re.sub(r"\n?```$", "", text.strip())
            return text
        except Exception as e:
            err_str = str(e)
            # Only fall through for access/availability errors
            if any(code in err_str for code in ("403", "404", "denied", "not found", "unavailable", "no longer available")):
                logger.warning(f"Gemini model '{model_name}' unavailable ({err_str[:120]}), trying next...")
                last_err = e
                continue
            # For other errors (bad request, rate limit, etc.) raise immediately
            logger.error(f"Gemini API error with model '{model_name}': {e}")
            raise

    logger.error(f"All Gemini models exhausted. Last error: {last_err}")
    raise last_err


async def ask_gemini_json(prompt: str, model: str = "flash") -> dict:
    """Ask Gemini and parse the response as JSON."""
    text = await ask_gemini(prompt, model=model, expect_json=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Gemini did not return valid JSON. Got: {text[:300]}")
