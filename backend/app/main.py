"""
FairLens Backend — FastAPI Application
Bias detection and remediation platform for Google Solution Challenge 2026.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from app.core.config import settings
from app.core.logging import setup_logging
from app.api import cartography, constitution, proxy_hunter, redteam, health

setup_logging()

app = FastAPI(
    title="FairLens API",
    description="AI Bias Detection & Remediation Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(cartography.router, prefix="/api/v1/cartography", tags=["bias-cartography"])
app.include_router(constitution.router, prefix="/api/v1/constitution", tags=["counterfactual-constitution"])
app.include_router(proxy_hunter.router, prefix="/api/v1/proxy", tags=["proxy-variable-hunter"])
app.include_router(redteam.router, prefix="/api/v1/redteam", tags=["fairness-redteam"])


@app.get("/")
async def root():
    return {
        "service": "FairLens API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
