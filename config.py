"""
Configuration handling for EVA backend using Pydantic-Settings.

This module configures all system settings, loading from environment
variables and Google Cloud Secrets where appropriate.
"""

import os
from typing import Optional, Dict, Any, List, Set

from pydantic import Field, field_validator, AnyHttpUrl, EmailStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists (useful for local dev)
load_dotenv()

class Settings(BaseSettings):
    """System settings loaded from environment variables and secrets."""

    # --- Environment & Application Info ---
    ENVIRONMENT: str = Field(default="development", description="Application environment (development, staging, production)")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    HOST: str = Field(default="0.0.0.0", description="Host address to bind the server")
    PORT: int = Field(default=8080, description="Port to bind the server")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    APP_NAME: str = Field(default="EVA Backend", description="Application name")
    APP_VERSION: str = Field(default="2.0.0", description="Application version")
    GOOGLE_CLOUD_PROJECT: Optional[str] = Field(default=None, description="Google Cloud Project ID")
    SERVICE_URL: Optional[AnyHttpUrl] = Field(default=None, description="Public URL of the Cloud Run service")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")

    # --- API Configuration ---
    API_VERSION: str = Field(default="2.0.0", description="API version string")
    API_TITLE: str = Field(default="EVA Backend API", description="API title for documentation")
    API_DESCRIPTION: str = Field(default="Enhanced Virtual Assistant Backend API", description="API description for documentation")

    # --- Authentication & Security ---
    # SECRET_KEY: Loaded from Secret Manager secret 'SECRET_KEY'
    SECRET_KEY: str = Field(..., description="Secret key for JWT signing (min 32 chars)")
    # MASTER_ENCRYPTION_KEY: Loaded from Secret Manager secret 'MASTER_ENCRYPTION_KEY'
    MASTER_ENCRYPTION_KEY: str = Field(..., description="Master key for encrypting secrets (Fernet key, base64 encoded)")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24, description="Access token validity duration in minutes") # 1 day
    ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    EVA_INITIAL_DEVICE_SECRET: Optional[str] = Field(default=None, description="Secret for initial device pairing")
    DEVICE_TOKEN_SECRET: Optional[str] = Field(default=None, description="Secret related to device tokens")

    # --- Database Configuration ---
    DB_PROVIDER: str = Field(default="firebase", description="Database provider ('firebase')")
    FIREBASE_CREDENTIALS_PATH: str = Field(default="/app/secrets/firebase-credentials.json", description="Path to Firebase service account key file")

    # --- LLM Service (Gemini) ---
    LLM_PROVIDER: str = Field(default="gemini", description="LLM provider ('gemini')")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="API Key for Gemini")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", description="Gemini model name to use")
    LLM_MAX_TOKENS: int = Field(default=8192, description="Default max tokens for LLM generation")
    LLM_TEMPERATURE: float = Field(default=0.7, description="Default temperature for LLM generation")

    # --- Memory System ---
    MEMORY_REFRESH_BATCH_SIZE: int = Field(default=5, description="Batch size for memory refresh")
    MEMORY_MAX_CORE_MEMORIES: int = Field(default=50, description="Max core memories to retrieve")
    MEMORY_MAX_EVENT_MEMORIES: int = Field(default=10, description="Max event memories to retrieve")
    MEMORY_IMPORTANCE_THRESHOLD: int = Field(default=5, description="Min importance for auto-extracted memories")
    CORE_MEMORY_IMPORTANCE_THRESHOLD: int = Field(default=5, description="Alias for memory importance threshold") # Added for memory_extractor.py compatibility

    # --- Context Window ---
    CONTEXT_MAX_TOKENS: int = Field(default=16000, description="Max tokens in LLM context window")
    CONTEXT_MAX_MESSAGES: int = Field(default=20, description="Max recent messages to keep before summarization attempt")
    SUMMARIZE_AFTER_TURNS: int = Field(default=10, description="Number of turns before attempting summarization")

    # --- Rate Limiting ---
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Default requests per minute for free tier")
    RATE_LIMIT_PER_DAY: int = Field(default=1000, description="Default requests per day for free tier")

    # --- External Service URLs (Example for Weather Tool) ---
    WEATHER_API_URL: Optional[str] = Field(default=None, description="Base URL for weather API (e.g., OpenWeatherMap)")
    WEATHER_API_KEY: Optional[str] = Field(default=None, description="API Key for weather service (Loaded from Secret Manager)") # Load from secrets

    # --- Feature Flags (Defaults, can be overridden by env/files) ---
    FEATURES: Dict[str, bool] = Field(default_factory=lambda: {
        "memory_system": True,
        "conversation_analysis": True, # Needed for memory extraction/summarization
        "knowledge_integration": False, # Placeholder
        "real_time_responses": True, # WebSocket streaming
        "function_calling": True, # For api_tools.py
        "sentiment_analysis": False, # Stage 4
        "adaptive_response": False, # Stage 4
        "wellness_tools": False, # Stage 5
        "offline_sync": True, # For api_sync.py
    }, description="Feature flags")

    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file=".env",          # Load .env file if present
        env_nested_delimiter='__', # Use __ for nested env vars (e.g., GEMINI_CONFIG__MODEL)
        case_sensitive=False,     # Environment variables are usually case-insensitive
        extra='ignore'            # Ignore extra fields from environment/files
    )

    # --- Derived Properties ---
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    # --- Validators ---
    @field_validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @field_validator("MASTER_ENCRYPTION_KEY")
    def validate_master_key(cls, v):
        import base64
        try:
            # Must be a valid Fernet key (URL-safe base64 encoded 32 bytes)
            key_bytes = base64.urlsafe_b64decode(v)
            if len(key_bytes) != 32:
                raise ValueError("MASTER_ENCRYPTION_KEY must be a URL-safe base64 encoded 32-byte key")
        except (TypeError, ValueError, Exception) as e:
            raise ValueError(f"Invalid MASTER_ENCRYPTION_KEY format: {e}")
        return v

    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("CORS_ORIGINS", mode='before')
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

# --- Singleton Instance ---
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        if 'DEBUG' not in os.environ:
             _settings.DEBUG = not _settings.is_production
        print(f"Settings loaded for environment: {_settings.ENVIRONMENT}")
        print(f"Debug mode: {_settings.DEBUG}")
        print(f"Firebase Credentials Path: {_settings.FIREBASE_CREDENTIALS_PATH}")
        if not _settings.SERVICE_URL and _settings.CORS_ORIGINS and _settings.CORS_ORIGINS[0] != "*":
             _settings.SERVICE_URL = _settings.CORS_ORIGINS[0]
    return _settings

# Example usage (optional, for testing)
if __name__ == "__main__":
    settings = get_settings()
    print("--- Loaded Settings ---")
    # ... (rest of print statements) ...
    print(f"Master Key Loaded: {'Yes' if settings.MASTER_ENCRYPTION_KEY else 'No'}")
    print("-----------------------")
