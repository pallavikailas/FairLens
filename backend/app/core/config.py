"""Core configuration using Pydantic settings — reads from .env or GCP Secret Manager."""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "FairLens"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # GCP
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "fairlens-gsc2026")
    GOOGLE_CLOUD_REGION: str = "us-central1"
    GCS_BUCKET_NAME: str = "fairlens-models"

    # Vertex AI / Gemini
    VERTEX_AI_LOCATION: str = "us-central1"
    GEMINI_MODEL: str = "gemini-1.5-pro"
    VERTEX_EMBEDDING_MODEL: str = "text-embedding-004"

    # BigQuery
    BIGQUERY_DATASET: str = "fairlens_audit"
    BIGQUERY_TABLE_AUDITS: str = "bias_audits"
    BIGQUERY_TABLE_DECISIONS: str = "decision_logs"

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://fairlens.web.app",
        "https://fairlens-frontend-*.run.app",
    ]

    # Red-Team Agent
    REDTEAM_MAX_ITERATIONS: int = 50
    REDTEAM_BATCH_SIZE: int = 100

    # Bias thresholds (industry standard)
    DEMOGRAPHIC_PARITY_THRESHOLD: float = 0.1   # SPD <= 0.1 is acceptable
    EQUALIZED_ODDS_THRESHOLD: float = 0.1
    DISPARATE_IMPACT_THRESHOLD: float = 0.8     # DI >= 0.8 is the 4/5ths rule

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
