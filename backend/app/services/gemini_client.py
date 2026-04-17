"""
Gemini client via Vertex AI (google-cloud-aiplatform).

Authentication:
  - Local dev:  run `gcloud auth application-default login`
  - Cloud Run:  service account with roles/aiplatform.user (auto-injected)

No API key required — uses Google Application Default Credentials.
"""
import logging
import json
import re

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import GoogleAPIError

from app.core.config import settings

logger = logging.getLogger(__name__)

vertexai.init(
    project=settings.GOOGLE_CLOUD_PROJECT,
    location=settings.VERTEX_AI_LOCATION,
)

_generation_config = GenerationConfig(temperature=0.2, max_output_tokens=8192)


async def ask_gemini(prompt: str, expect_json: bool = False) -> str:
    """Send a prompt to Gemini 2.5 Flash via Vertex AI and return the text."""
    try:
        model = GenerativeModel(
            settings.GEMINI_MODEL,
            generation_config=_generation_config,
        )
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
