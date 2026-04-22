"""
Gemini client — supports two modes:
  1. Local dev: GEMINI_API_KEY in .env → uses google-generativeai directly (no GCP needed)
  2. Cloud Run:  no key needed → uses Vertex AI with service account ADC
"""
import asyncio
import logging
import json
import re

from app.core.config import settings

logger = logging.getLogger(__name__)


def _make_model():
    """Return a Gemini model using whichever auth is available."""
    if settings.GEMINI_API_KEY:
        import warnings
        import google.generativeai as genai
        from google.generativeai import GenerationConfig as GC
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            genai.configure(api_key=settings.GEMINI_API_KEY)
        return genai.GenerativeModel(
            settings.GEMINI_MODEL,
            generation_config=GC(temperature=0.2, max_output_tokens=8192),
        ), "genai"
    else:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
        vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.VERTEX_AI_LOCATION)
        return GenerativeModel(
            settings.GEMINI_MODEL,
            generation_config=GenerationConfig(temperature=0.2, max_output_tokens=8192),
        ), "vertexai"


async def ask_gemini(prompt: str, expect_json: bool = False) -> str:
    try:
        model, mode = _make_model()

        if mode == "genai":
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: model.generate_content(prompt)),
                timeout=55.0,
            )
        else:
            response = await asyncio.wait_for(
                model.generate_content_async(prompt),
                timeout=55.0,
            )

        text = response.text
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini timed out after 55 seconds")
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise

    if expect_json:
        text = re.sub(r"^```(?:json)?\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text.strip())

    return text


async def ask_gemini_json(prompt: str) -> dict:
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
