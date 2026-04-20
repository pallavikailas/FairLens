"""FairLens config — reads from environment variables."""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "FairLens"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # GCP
    GOOGLE_CLOUD_PROJECT: str = "fairlens-493318"
    GOOGLE_CLOUD_REGION: str = "us-central1"

    # Vertex AI — used for Gemini 2.5 Flash and text embeddings
    VERTEX_AI_LOCATION: str = "us-central1"
    VERTEX_EMBEDDING_MODEL: str = "text-embedding-004"
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Red-team agent
    REDTEAM_MAX_ITERATIONS: int = 3
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
        extra = "ignore"


settings = Settings()
