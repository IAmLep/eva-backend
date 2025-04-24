"""
Application Configuration using Pydantic Settings.

Loads configuration from environment variables and optional .env files.
Provides a centralized Settings object for the application.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# --- .env File Detection ---
PRIMARY_ENV_FILE = Path(__file__).parent / ".env"
SECONDARY_ENV_FILE = "project.env"

env_files = []
if PRIMARY_ENV_FILE.is_file():
    env_files.append(str(PRIMARY_ENV_FILE))
    logger.info(f"Found primary .env file: {PRIMARY_ENV_FILE}")
else:
    logger.warning(f"Primary .env file not found at: {PRIMARY_ENV_FILE}")

# Always include secondary (pydantic will ignore if missing)
env_files.append(SECONDARY_ENV_FILE)

# --- Settings Model ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=tuple(env_files),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # — Application Info —
    PROJECT_NAME: str = "EVA Backend"
    API_TITLE: str = "EVA API"
    API_DESCRIPTION: str = "Backend services for the EVA project."
    API_VERSION: str = "0.1.0"

    # — Server & Debug —
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # — Production Flag (for security.py) —
    is_production: bool = False

    # — Firebase / Firestore —
    PROJECT_ID: str = "default-project-id"
    FIREBASE_PROJECT_ID: str = "default-firebase-project-id"
    USE_GCP_DEFAULT_CREDENTIALS: bool = True
    FIREBASE_CREDENTIALS_PATH: Optional[str] = None

    # — External API Keys —
    GEMINI_API_KEY: Optional[str] = None

    # — Rate Limiting —
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_USER_REQUESTS: int = 100
    RATE_LIMIT_USER_WINDOW_SECONDS: int = 60

    # — CORS —
    CORS_ORIGINS: str = "*"  
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # — Security / Auth —
    SECRET_KEY: str = "replace_me_with_secure_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

try:
    settings = Settings()
    logger.info("Configuration loaded successfully.")
    logger.debug(f"Settings snapshot: {settings.dict()}")
except Exception as e:
    logger.exception("Failed to load configuration", exc_info=e)
    raise ConfigurationError(f"Configuration loading failed: {e}") from e