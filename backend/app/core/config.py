"""
FairLens Config — reads from environment variables and GCP Secret Manager.
All fields have safe defaults so the app never crashes on missing optional settings.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "FairLens"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # GCP
    GOOGLE_CLOUD_PROJECT: str = "fairlens-493318"
    GOOGLE_CLOUD_REGION: str = "us-central1"
    GCS_BUCKET_NAME: str = "fairlens-models"

    # Vertex AI
    VERTEX_AI_LOCATION: str = "us-central1"
    VERTEX_EMBEDDING_MODEL: str = "text-embedding-004"

    # Gemini — value injected at runtime from GCP Secret Manager
    # Secret name in GCP: fairlens-gemini-key
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_FLASH_MODEL: str = "gemini-2.0-flash"

    # BigQuery
    BIGQUERY_DATASET: str = "fairlens_audit"
    BIGQUERY_TABLE_AUDITS: str = "bias_audits"
    BIGQUERY_TABLE_DECISIONS: str = "decision_logs"

    # Red-team agent
    REDTEAM_MAX_ITERATIONS: int = 50
    REDTEAM_BATCH_SIZE: int = 100

    # Fairness thresholds (industry standard)
    DEMOGRAPHIC_PARITY_THRESHOLD: float = 0.1
    DISPARATE_IMPACT_THRESHOLD: float = 0.8

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://fairlens.web.app",
        "https://fairlens-frontend-nrk2z2yadq-uc.a.run.app",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"   # never crash on unexpected env vars


settings = Settings()
