import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List, Dict
from google.cloud import secretmanager
import secrets
from pathlib import Path
from functools import lru_cache

load_dotenv()

# Environment setup
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Secret management
def get_secret(secret_id, default_value=None):
    """Get secret from Secret Manager or environment variables with robust fallback"""
    if IS_PRODUCTION:
        try:
            client = secretmanager.SecretManagerServiceClient()
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                print(f"WARNING: GOOGLE_CLOUD_PROJECT not set, using hardcoded project ID")
                project_id = "eva-ai-454545"  # Hardcode your project ID as backup
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(name=name)
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"Error retrieving secret {secret_id}: {e}")
            # Fall back to environment variables
            value = os.getenv(secret_id)
            if value:
                return value
            # If still not found, use the default value
            print(f"WARNING: Using default value for {secret_id}")
            return default_value
    else:
        # In development, use environment variables with default fallback
        value = os.getenv(secret_id)
        return value if value is not None else default_value

# Generate a persistent SECRET_KEY or use environment variable
# This single definition replaces the duplicate definitions
SECRET_KEY = get_secret("SECRET_KEY", os.urandom(24).hex())
if SECRET_KEY == os.urandom(24).hex():  # If we're using the generated fallback
    print("WARNING: Using temporary SECRET_KEY. Set SECRET_KEY in Cloud Run environment variables for persistence.")

# API Keys with default empty string to prevent None errors
GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "")

# App settings
APP_NAME = "Eva LLM Application"
DEVICE_TOKEN = get_secret("DEVICE_TOKEN", os.urandom(8).hex())  # For device authentication

# Database settings
if IS_PRODUCTION:
    # Look for DATABASE_URL as environment variable first
    DB_CONNECTION_STRING = os.getenv("DATABASE_URL", "sqlite:////mnt/eva-memory/eva.db")
    if not DB_CONNECTION_STRING:
        # Fall back to secret if env var not found
        DB_CONNECTION_STRING = get_secret("DB_CONNECTION_STRING", "sqlite:////mnt/eva-memory/eva.db")
else:
    DB_CONNECTION_STRING = "sqlite:///./sqlite.db"

# Redis settings with robust defaults
if IS_PRODUCTION:
    REDIS_HOST = get_secret("REDIS_HOST", os.getenv("REDIS_HOST", "localhost"))
    REDIS_PORT = int(get_secret("REDIS_PORT", os.getenv("REDIS_PORT", "6379")))
    REDIS_PASSWORD = get_secret("REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", ""))
else:
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_PASSWORD = ""

# mTLS settings
MTLS_ENABLED = IS_PRODUCTION or os.getenv("ENABLE_MTLS", "false").lower() == "true"
CLIENT_CERT_PATH = os.getenv("CLIENT_CERT_PATH", "./certs/client.crt")
CLIENT_KEY_PATH = os.getenv("CLIENT_KEY_PATH", "./certs/client.key")
CA_CERT_PATH = os.getenv("CA_CERT_PATH", "./certs/ca.crt")

# WebSocket settings
WS_HEARTBEAT_INTERVAL = 30  # seconds

# Token configuration
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 30))

# OAuth client settings - in production, store these in a database or Secret Manager
OAUTH_CLIENTS = {
    "default": {
        "client_id": os.getenv("DEFAULT_CLIENT_ID", "default-client"),
        "client_secret": os.getenv("DEFAULT_CLIENT_SECRET", "default-secret"),
        "redirect_uris": os.getenv("DEFAULT_REDIRECT_URIS", "").split(","),
        "scopes": ["chat:read", "chat:write", "profile:read"]
    }
}

class Settings(BaseSettings):
    """Configuration settings for the application."""

    # Database settings
    DATABASE_URL: str = Field(default=DB_CONNECTION_STRING)
    
    # Redis
    REDIS_HOST: str = Field(default=REDIS_HOST)
    REDIS_PORT: int = Field(default=REDIS_PORT)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: str = Field(default=REDIS_PASSWORD or "")  # Empty string instead of None
    REDIS_RETRY_ON_STARTUP: bool = Field(default=True)
    REDIS_MAX_RETRIES: int = Field(default=5)
    REDIS_RETRY_INTERVAL: int = Field(default=2)  # seconds
    
    # Gemini AI - allow empty string to prevent validation errors
    GEMINI_API_KEY: str = Field(default=GEMINI_API_KEY)
    GEMINI_MODEL: str = Field(default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    GEMINI_MAX_TOKENS: int = Field(default=int(os.getenv("GEMINI_MAX_TOKENS", "150")))
    GEMINI_TEMPERATURE: float = Field(default=float(os.getenv("GEMINI_TEMPERATURE", "0.7")))
    GEMINI_TOP_P: float = Field(default=float(os.getenv("GEMINI_TOP_P", "1.0")))
    
    # Rate Limiting
    RATELIMIT_PER_MINUTE: int = Field(default=60)
    RATELIMIT_ENABLED: bool = Field(default=True)
    RATELIMIT_STORAGE_URI: str = Field(default="memory://")
    
    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8080)
    DEBUG: bool = Field(default=not IS_PRODUCTION)  # Set based on environment
    CORS_ORIGINS: List[str] = Field(default=["*"])
    LOG_LEVEL: str = Field(default="info")
    
    # Authentication
    SECRET_KEY: str = Field(default=SECRET_KEY)
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_DAYS: int = Field(default=30)
    VERIFICATION_TIMEOUT: int = Field(default=300)
    
    # Memory Management
    MEMORY_CLEANUP_INTERVAL: int = Field(default=3600)  # seconds
    MAX_MEMORY_AGE: int = Field(default=86400)  # seconds
    
    # OAuth2 Settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30)
    OAUTH2_CLIENT_ID: str = Field(default=os.getenv("OAUTH2_CLIENT_ID", "evacore-client"))
    OAUTH2_CLIENT_SECRET: str = Field(
        default=os.getenv("OAUTH2_CLIENT_SECRET", secrets.token_hex(24))
    )
    OAUTH2_SCOPES: Dict[str, str] = Field(default={
        "chat:read": "Read access to chat data",
        "chat:write": "Write access to chat data"
    })
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = Settings()
# Export settings as config for backward compatibility
config = settings

__all__ = ['settings', 'config']

# Add this to test the settings
if __name__ == "__main__":
    print("Settings loaded successfully:")
    print(f"Host: {settings.HOST}")
    print(f"Port: {settings.PORT}")
    print(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    print(f"Gemini API Key: {'[NOT SET]' if not settings.GEMINI_API_KEY else settings.GEMINI_API_KEY[:5] + '...'}")
    print(f"Environment: {'Production' if IS_PRODUCTION else 'Development'}")