"""Health and readiness endpoints for Cloud Run."""
from fastapi import APIRouter
from datetime import datetime
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready")
async def ready():
    return {"status": "ready"}


@router.get("/gemini")
async def gemini_diagnostic():
    """Test Gemini connectivity. Visit /health/gemini to debug API key issues."""
    if not settings.GEMINI_API_KEY:
        return {
            "key_set": False,
            "error": "GEMINI_API_KEY is not set",
            "fix": "Set GEMINI_API_KEY in your .env or Cloud Run env vars",
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = await model.generate_content_async("Say OK")
        return {
            "key_set": True,
            "key_prefix": settings.GEMINI_API_KEY[:8] + "...",
            "model": "gemini-2.5-flash",
            "status": "ok",
            "response": response.text.strip(),
        }
    except Exception as e:
        return {
            "key_set": True,
            "key_prefix": settings.GEMINI_API_KEY[:8] + "...",
            "status": "error",
            "error": str(e),
        }
