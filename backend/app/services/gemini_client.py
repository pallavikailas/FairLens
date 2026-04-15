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

_pro   = None
_flash = None

def get_pro():
    global _pro
    if _pro is None:
        _pro = genai.GenerativeModel(settings.GEMINI_MODEL)
    return _pro

def get_flash():
    global _flash
    if _flash is None:
        _flash = genai.GenerativeModel(settings.GEMINI_FLASH_MODEL)
    return _flash


async def ask_gemini(prompt: str, model: str = "pro", expect_json: bool = False) -> str:
    """Send a prompt to Gemini and return the text response."""
    m = get_pro() if model == "pro" else get_flash()
    try:
        response = m.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192,
            )
        )
        text = response.text
        if expect_json:
            # Strip markdown fences if present
            text = re.sub(r"^```(?:json)?\n?", "", text.strip())
            text = re.sub(r"\n?```$", "", text.strip())
        return text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise


async def ask_gemini_json(prompt: str, model: str = "flash") -> dict:
    """Ask Gemini and parse the response as JSON."""
    text = await ask_gemini(prompt, model=model, expect_json=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Gemini did not return valid JSON. Got: {text[:300]}")
