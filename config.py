"""
Configuration Management for EVA Backend.

Loads settings from environment variables and .env files using pydantic-settings.
Provides a centralized way to access configuration values throughout the application.
"""

import logging
import os
from functools import lru_cache
from typing import List, Optional, Union # Keep List/Optional/Union for other fields

# field_validator is no longer needed for CORS_ORIGINS
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variable Handling ---
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


# --- Settings Model ---
class Settings(BaseSettings):
    """
    Defines application settings, loaded from environment variables and .env files.
    """
    # --- Core Application Settings ---
    APP_NAME: str = "EVA Backend"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = Field(default=APP_ENV)
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    API_TITLE: str = Field(default="EVA API", description="Swagger UI Title")
    API_DESCRIPTION: str = Field(default="Backend services for the EVA project", description="Swagger UI Description")
    API_VERSION: str = Field(default="0.1.0", description="API version")

    # --- API / Server Settings ---
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = Field(default="EVA", description="Project Name")
    PORT: int = Field(default=8080, description="Port")
    HOST: str = Field(default="0.0.0.0", description="Host")

    # --- Security Settings ---
    SECRET_KEY: str = Field(..., description="Secret key for JWT")
    ALGORITHM: str = Field(default="HS256", description="JWT Algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24 * 7, description="Access token expiry (minutes)")

    # --- CORS Settings ---
    # CHANGE: Read CORS_ORIGINS as a plain string from the environment
    CORS_ORIGINS: str = Field(default="*", description="Comma-separated string of allowed CORS origins")
    # Keep other CORS settings as they were
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])

    # --- Database Settings ---
    FIREBASE_PROJECT_ID: Optional[str] = Field(default=None, description="Firebase Project ID")
    FIREBASE_CREDENTIALS_PATH: Optional[str] = Field(default="/app/secrets/firebase-credentials.json", description="Path to Firebase credentials")
    USE_GCP_DEFAULT_CREDENTIALS: bool = Field(default=False, description="Use GCP default credentials")

    # --- LLM Service Settings ---
    LLM_PROVIDER: str = Field(default="gemini", description="LLM provider ('gemini', 'openai')")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Gemini API Key")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash-latest", description="Gemini model")
    # OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    # OPENAI_MODEL: str = Field(default="gpt-4o", description="OpenAI model")

    # --- LLM Generation Parameters ---
    LLM_TEMPERATURE: float = Field(default=0.7, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(default=2048, description="LLM max tokens")

    # --- Rate Limiting ---
    RATE_LIMIT_USER_REQUESTS: int = Field(default=100, description="Rate limit user requests")
    RATE_LIMIT_USER_WINDOW_SECONDS: int = Field(default=60, description="Rate limit user window (s)")
    RATE_LIMIT_GLOBAL_REQUESTS: int = Field(default=1000, description="Rate limit global requests")
    RATE_LIMIT_GLOBAL_WINDOW_SECONDS: int = Field(default=60, description="Rate limit global window (s)")
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")

    # --- WebSocket Settings ---
    WEBSOCKET_MAX_QUEUE_SIZE: int = Field(default=10, description="WS max queue size")
    WEBSOCKET_TIMEOUT_SECONDS: int = Field(default=60 * 10, description="WS timeout (s)")

    # --- Memory Management ---
    MEMORY_DEFAULT_RETENTION_DAYS: Optional[int] = Field(default=None, description="Memory retention (days)")
    MEMORY_SUMMARY_THRESHOLD: int = Field(default=10, description="Memory summary threshold")
    MEMORY_SEARCH_LIMIT: int = Field(default=10, description="Memory search limit")

    # --- Caching ---
    CACHE_ENABLED: bool = Field(default=True, description="Enable caching")
    CACHE_DEFAULT_TTL_SECONDS: int = Field(default=300, description="Cache TTL (s)")

    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False
    )

# --- Singleton Instance ---
@lru_cache()
def get_settings() -> Settings:
    """Returns the cached Settings instance."""
    logger.info("Initializing application settings...")
    try:
        _settings = Settings()
        logger.info(f"APP_ENV: {_settings.APP_ENV}")
        logger.info(f"LLM_PROVIDER: {_settings.LLM_PROVIDER}")
        logger.info(f"GEMINI_MODEL: {_settings.GEMINI_MODEL}")
        # Log the raw string value loaded from the environment
        logger.info(f"CORS_ORIGINS (raw string): {_settings.CORS_ORIGINS}")
        return _settings
    except Exception as e:
        logger.exception("CRITICAL: Failed to load application settings.", exc_info=e)
        raise RuntimeError(f"Failed to load application settings: {e}")


# --- Example Usage (for testing if run directly) ---
if __name__ == "__main__":
    settings = get_settings()
    print("Settings loaded successfully:")
    print(f"  App Env: {settings.APP_ENV}")
    print(f"  CORS Origins (raw): {settings.CORS_ORIGINS}")
    # You could add more fields here for testing