"""Health and readiness endpoints for Cloud Run."""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@router.get("/ready")
async def ready():
    return {"status": "ready"}
