"""
Gemini client — supports two modes:
  1. Local dev: GEMINI_API_KEY in .env → uses google-genai directly (no GCP needed)
  2. Cloud Run:  no key needed → uses Vertex AI with service account ADC
"""
import asyncio
import logging
import json
import re

from app.core.config import settings

logger = logging.getLogger(__name__)

_GEN_CONFIG = {"temperature": 0.2, "max_output_tokens": 8192}


def _make_client():
    """Return a (client, model_name) tuple using whichever auth is available."""
    from google import genai

    if settings.GEMINI_API_KEY:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return client, settings.GEMINI_MODEL, "genai"
    else:
        client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.VERTEX_AI_LOCATION,
        )
        return client, settings.GEMINI_MODEL, "vertexai"


async def ask_gemini(prompt: str, expect_json: bool = False) -> str:
    from google.genai import types

    client, model_name, mode = _make_client()

    gen_config = types.GenerateContentConfig(**_GEN_CONFIG)

    try:
        if mode == "genai":
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=gen_config,
                    ),
                ),
                timeout=55.0,
            )
        else:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=gen_config,
                ),
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
