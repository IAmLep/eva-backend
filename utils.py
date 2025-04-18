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
    # ENVIRONMENT: Set via Cloud Run env var (e.g., "production", "development")
    ENVIRONMENT: str = Field(default="development", description="Application environment (development, staging, production)")
    # DEBUG: Set based on ENVIRONMENT or explicitly
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    # HOST: Used by Uvicorn, typically 0.0.0.0 in containers
    HOST: str = Field(default="0.0.0.0", description="Host address to bind the server")
    # PORT: Set by Cloud Run automatically, or 8080 locally
    PORT: int = Field(default=8080, description="Port to bind the server")
    # LOG_LEVEL: Set via Cloud Run env var LOG_LEVEL (e.g., "INFO", "DEBUG")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    APP_NAME: str = Field(default="EVA Backend", description="Application name")
    APP_VERSION: str = Field(default="2.0.0", description="Application version")
    # GOOGLE_CLOUD_PROJECT: Set via Cloud Run env var
    GOOGLE_CLOUD_PROJECT: Optional[str] = Field(default=None, description="Google Cloud Project ID")
    # SERVICE_URL: Set via Cloud Run env var CORS_ORIGINS or inferred
    SERVICE_URL: Optional[AnyHttpUrl] = Field(default=None, description="Public URL of the Cloud Run service")
    # CORS_ORIGINS: Set via Cloud Run env var (comma-separated string or JSON list)
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")

    # --- API Configuration ---
    API_VERSION: str = Field(default="2.0.0", description="API version string")
    API_TITLE: str = Field(default="EVA Backend API", description="API title for documentation")
    API_DESCRIPTION: str = Field(default="Enhanced Virtual Assistant Backend API", description="API description for documentation")

    # --- Authentication & Security ---
    # SECRET_KEY: Loaded from Secret Manager secret 'SECRET_KEY'
    SECRET_KEY: str = Field(..., description="Secret key for JWT and encryption (min 32 chars)")
    # ACCESS_TOKEN_EXPIRE_MINUTES: How long JWT tokens are valid
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24, description="Access token validity duration in minutes") # 1 day
    # ALGORITHM: JWT signing algorithm
    ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    # EVA_INITIAL_DEVICE_SECRET: Set via Cloud Run env var (if needed for initial device setup)
    EVA_INITIAL_DEVICE_SECRET: Optional[str] = Field(default=None, description="Secret for initial device pairing")
    # DEVICE_TOKEN_SECRET: Loaded from Secret Manager secret 'DEVICE_TOKEN' (Purpose unclear from context, added for completeness)
    DEVICE_TOKEN_SECRET: Optional[str] = Field(default=None, description="Secret related to device tokens")

    # --- Database Configuration ---
    DB_PROVIDER: str = Field(default="firebase", description="Database provider ('firebase')")
    # FIREBASE_CREDENTIALS_PATH: Path where the ACCOUNT_KEY secret is mounted
    FIREBASE_CREDENTIALS_PATH: str = Field(default="/app/secrets/firebase-credentials.json", description="Path to Firebase service account key file")

    # --- LLM Service (Gemini) ---
    LLM_PROVIDER: str = Field(default="gemini", description="LLM provider ('gemini')")
    # GEMINI_API_KEY: Loaded from Secret Manager secret 'GEMINI_API_KEY'
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="API Key for Gemini")
    # GEMINI_MODEL: Set via Cloud Run env var
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

    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("CORS_ORIGINS", mode='before')
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            # If it's a comma-separated string from env var
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return v
        # Default or handle other types if necessary
        return ["*"] # Fallback to allow all if misconfigured

# --- Singleton Instance ---
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """
    Return the singleton Settings instance, loading it if necessary.
    """
    global _settings
    if _settings is None:
        # Pass _env_file=None to prevent double loading if .env is already handled by load_dotenv
        # Pydantic-settings v2 automatically finds .env
        _settings = Settings()
        # Set DEBUG based on environment if not explicitly set
        if 'DEBUG' not in os.environ:
             _settings.DEBUG = not _settings.is_production
        print(f"Settings loaded for environment: {_settings.ENVIRONMENT}") # Add print for verification
        print(f"Debug mode: {_settings.DEBUG}")
        print(f"Firebase Credentials Path: {_settings.FIREBASE_CREDENTIALS_PATH}")
        # Manually set SERVICE_URL from CORS_ORIGINS if not set
        if not _settings.SERVICE_URL and _settings.CORS_ORIGINS and _settings.CORS_ORIGINS[0] != "*":
             _settings.SERVICE_URL = _settings.CORS_ORIGINS[0]

    return _settings

# Example usage (optional, for testing)
if __name__ == "__main__":
    settings = get_settings()
    print("--- Loaded Settings ---")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"Project ID: {settings.GOOGLE_CLOUD_PROJECT}")
    print(f"Service URL: {settings.SERVICE_URL}")
    print(f"CORS Origins: {settings.CORS_ORIGINS}")
    print(f"Secret Key Loaded: {'Yes' if settings.SECRET_KEY else 'No'}")
    print(f"Gemini API Key Loaded: {'Yes' if settings.GEMINI_API_KEY else 'No'}")
    print(f"Firebase Credentials Path: {settings.FIREBASE_CREDENTIALS_PATH}")
    print(f"Features: {settings.FEATURES}")
    print(f"Is Production: {settings.is_production}")
    print("-----------------------")
