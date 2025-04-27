"""
Configuration Management for EVA Backend.

Loads settings from environment variables and .env files using pydantic‑settings.
Provides a centralized way to access configuration values throughout the application.
"""
import logging
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Determine which .env file to load ---
APP_ENV = os.getenv("APP_ENV", "development").lower()
logger.info(f"Application environment detected: {APP_ENV}")

env_file = f".env.{APP_ENV}" if APP_ENV != "production" else ".env"
if not os.path.exists(env_file) and APP_ENV != "production":
    logger.warning(f"Environment file '{env_file}' not found. Trying default '.env'.")
    env_file = ".env"
if os.path.exists(env_file):
    logger.info(f"Loading settings from environment file: {env_file}")
else:
    logger.warning(f"Environment file '{env_file}' not found. Relying solely on environment variables.")
    env_file = None

class Settings(BaseSettings):
    """Defines application settings, loaded from environment variables and .env files."""
    # --- Core Application Settings ---
    APP_NAME: str = "EVA Backend"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = Field(default=APP_ENV)
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    API_TITLE: str = Field(default="EVA API")
    API_DESCRIPTION: str = Field(default="Backend services for the EVA project")
    API_VERSION: str = Field(default="0.1.0")

    # --- API / Server Settings ---
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = Field(default="EVA Backend")
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8080)

    # --- Security & Auth Settings ---
    SECRET_KEY: str = Field(..., description="Secret key for internal JWT")
    ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24 * 7, description="Token expiry (minutes)")
    BACKEND_URL: Optional[str] = Field(
        default=None,
        description="Cloud Run service URL, used as audience for Google ID‑tokens."
    )
    API_AUDIENCE: Optional[str] = Field(default=None, description="HS256 token audience (if any)")
    is_production: bool = Field(default=False, description="Enable production-only features (HSTS)")

    # --- CORS Settings ---
    CORS_ORIGINS: str = Field(default="*", description="Comma-separated CORS origins")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])

    # --- Database / LLM / Rate Limiting / WebSockets / Caching ---
    FIREBASE_PROJECT_ID: Optional[str] = None
    FIREBASE_CREDENTIALS_PATH: Optional[str] = "/app/secrets/firebase-credentials.json"
    USE_GCP_DEFAULT_CREDENTIALS: bool = False
    LLM_PROVIDER: str = Field(default="gemini")
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash-latest")
    LLM_TEMPERATURE: float = Field(default=0.7)
    LLM_MAX_TOKENS: int = Field(default=2048)
    RATE_LIMIT_USER_REQUESTS: int = Field(default=100)
    RATE_LIMIT_USER_WINDOW_SECONDS: int = Field(default=60)
    RATE_LIMIT_GLOBAL_REQUESTS: int = Field(default=1000)
    RATE_LIMIT_GLOBAL_WINDOW_SECONDS: int = Field(default=60)
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    WEBSOCKET_MAX_QUEUE_SIZE: int = Field(default=10)
    WEBSOCKET_TIMEOUT_SECONDS: int = Field(default=600)
    MEMORY_DEFAULT_RETENTION_DAYS: Optional[int] = Field(default=None)
    MEMORY_SUMMARY_THRESHOLD: int = Field(default=10)
    MEMORY_SEARCH_LIMIT: int = Field(default=10)
    CACHE_ENABLED: bool = Field(default=True)
    CACHE_DEFAULT_TTL_SECONDS: int = Field(default=300)

    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance, applying sane defaults if needed."""
    try:
        s = Settings()
    except Exception as e:
        logger.error("Failed to load application settings", exc_info=e)
        raise

    if not s.BACKEND_URL:
        # Default Cloud Run URL for local/dev
        default_url = f"http://{s.HOST}:{s.PORT}"
        logger.warning(f"BACKEND_URL not set; defaulting to {default_url}")
        s.BACKEND_URL = default_url

    logger.info(f"Settings loaded: ENV={s.APP_ENV}, BACKEND_URL={s.BACKEND_URL}")
    return s

# Module‐level singleton for easy import
settings = get_settings()