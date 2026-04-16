"""Health and readiness endpoints for Cloud Run."""
from fastapi import APIRouter
from datetime import datetime
import httpx
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
    """
    Diagnoses the Gemini API key by calling ListModels.
    Visit /health/gemini in your browser to debug API key issues.
    """
    key = settings.GEMINI_API_KEY
    if not key:
        return {"error": "GEMINI_API_KEY is not set in environment", "fix": "Set GEMINI_API_KEY in Cloud Run env vars or Secret Manager"}

    results = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Check what models are available with this key
        for api_version in ("v1", "v1beta"):
            url = f"https://generativelanguage.googleapis.com/{api_version}/models"
            resp = await client.get(url, params={"key": key})
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                results[api_version] = {"status": "ok", "model_count": len(models), "models": models[:20]}
            else:
                results[api_version] = {"status": f"HTTP {resp.status_code}", "body": resp.text[:300]}

        # Try a minimal generate call with the first working model
        test_result = "not attempted"
        for api_version in ("v1", "v1beta"):
            if results.get(api_version, {}).get("status") == "ok":
                model_names = results[api_version]["models"]
                generate_candidates = [m.replace("models/", "") for m in model_names if "flash" in m]
                if generate_candidates:
                    test_model = generate_candidates[0]
                    test_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{test_model}:generateContent"
                    test_resp = await client.post(
                        test_url,
                        json={"contents": [{"parts": [{"text": "Say OK"}]}]},
                        params={"key": key},
                        headers={"Content-Type": "application/json"},
                    )
                    test_result = f"HTTP {test_resp.status_code} using {api_version}/{test_model}"
                    if test_resp.status_code == 200:
                        test_result += " ✓ WORKING"
                    break

    return {
        "key_set": True,
        "key_prefix": key[:8] + "...",
        "list_models": results,
        "generate_test": test_result,
    }
