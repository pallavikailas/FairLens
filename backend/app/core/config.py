"""Core configuration — reads entirely from environment variables / GCP secrets."""
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

    # Gemini — injected from GCP Secret Manager at runtime
    # Secret name: fairlens-gemini-key
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-pro"
    GEMINI_FLASH_MODEL: str = "gemini-1.5-flash"

    # BigQuery
    BIGQUERY_DATASET: str = "fairlens_audit"
    BIGQUERY_TABLE_AUDITS: str = "bias_audits"

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://fairlens.web.app",
    ]

    # Fairness thresholds
    DEMOGRAPHIC_PARITY_THRESHOLD: float = 0.1
    DISPARATE_IMPACT_THRESHOLD: float = 0.8

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
