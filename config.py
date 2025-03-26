"""
Configuration settings for the Eva LLM Application.
"""
import os
import sys
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List, Dict
from google.cloud import secretmanager
import secrets
from functools import lru_cache
import traceback

load_dotenv()

# Environment setup
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"
print(f"Running in {'production' if IS_PRODUCTION else 'development'} mode")

# Secret management - prioritize environment variables
def get_secret(secret_id, default_value=None):
    """Get secret preferably from environment variables, fallback to Secret Manager."""
    # Always check environment variables first
    env_value = os.getenv(secret_id)
    if env_value is not None:
        return env_value
        
    # Only look in Secret Manager for truly sensitive secrets in production
    if IS_PRODUCTION and secret_id in ["GEMINI_API_KEY", "SECRET_KEY", "OPENAI_API_KEY"]:
        try:
            client = secretmanager.SecretManagerServiceClient()
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                print(f"WARNING: GOOGLE_CLOUD_PROJECT not set, using hardcoded project ID")
                project_id = "eva-ai-454545"  # Using your project ID
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(name=name)
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"Error retrieving secret {secret_id}: {e}")
    
    return default_value

# Generate a persistent SECRET_KEY or use environment variable
SECRET_KEY = get_secret("SECRET_KEY", secrets.token_hex(32))

# API Keys
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")  # Added OpenAI API key

# App settings
APP_NAME = "Eva AI Assistant"
DEVICE_TOKEN = os.getenv("DEVICE_TOKEN", secrets.token_hex(8))

# Database settings
DB_CONNECTION_STRING = os.getenv("DATABASE_URL", os.getenv("database_url", "sqlite:///./sqlite.db"))
if IS_PRODUCTION and not DB_CONNECTION_STRING:
    DB_CONNECTION_STRING = "sqlite:////mnt/eva-memory/eva.db"

# Redis settings - ONLY use environment variables, never secrets
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
# Fixed typo in env var name
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", os.getenv("REDIS_PASSOWRD", ""))

# WebSocket settings
WS_HEARTBEAT_INTERVAL = 30  # seconds

# Token configuration
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
JWT_ALGORITHM = "HS256"

# OAuth2 client configuration
OAUTH2_CLIENT_ID = os.getenv("OAUTH2_CLIENT_ID", "evacore-client")
OAUTH2_CLIENT_SECRET = os.getenv("OAUTH2_CLIENT_SECRET", secrets.token_hex(24))

# OAuth2 registered clients
OAUTH_CLIENTS = {
    OAUTH2_CLIENT_ID: {
        "client_id": OAUTH2_CLIENT_ID,
        "client_secret": OAUTH2_CLIENT_SECRET,
        "client_name": "Eva Core Client",
        "redirect_uris": [
            "https://eva-ai-app.web.app/auth/callback",
            "http://localhost:3000/auth/callback",
            "app://eva-auth-callback"
        ],
        "grant_types": ["authorization_code", "refresh_token", "password"],
        "response_types": ["code"],
        "scope": "chat:read chat:write profile:read",
        "token_endpoint_auth_method": "client_secret_basic"
    }
}

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "openai"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

class Settings(BaseSettings):
    """Configuration settings for the application."""
    # Database settings
    DATABASE_URL: str = Field(default=DB_CONNECTION_STRING)
    
    # Redis settings
    REDIS_HOST: str = Field(default=REDIS_HOST)
    REDIS_PORT: int = Field(default=REDIS_PORT)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: str = Field(default=REDIS_PASSWORD)
    REDIS_RETRY_ON_STARTUP: bool = Field(default=True)
    REDIS_MAX_RETRIES: int = Field(default=5)
    REDIS_RETRY_INTERVAL: int = Field(default=2)  # seconds
    
    # Gemini AI
    GEMINI_API_KEY: Optional[str] = Field(default=GEMINI_API_KEY)
    GEMINI_MODEL: str = Field(default=GEMINI_MODEL)
    GEMINI_MAX_TOKENS: int = Field(default=int(os.getenv("GEMINI_MAX_TOKENS", "150")))
    GEMINI_TEMPERATURE: float = Field(default=float(os.getenv("GEMINI_TEMPERATURE", "0.7")))
    GEMINI_TOP_P: float = Field(default=float(os.getenv("GEMINI_TOP_P", "1.0")))
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=OPENAI_API_KEY)
    OPENAI_MODEL: str = Field(default=OPENAI_MODEL)
    OPENAI_MAX_TOKENS: int = Field(default=int(os.getenv("OPENAI_MAX_TOKENS", "150")))
    OPENAI_TEMPERATURE: float = Field(default=float(os.getenv("OPENAI_TEMPERATURE", "0.7")))
    
    # LLM settings
    LLM_PROVIDER: str = Field(default=LLM_PROVIDER)
    
    # Rate Limiting
    RATELIMIT_PER_MINUTE: int = Field(default=60)
    RATELIMIT_ENABLED: bool = Field(default=True)
    RATELIMIT_STORAGE_URI: str = Field(default="memory://")
    
    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8080)
    DEBUG: bool = Field(default=not IS_PRODUCTION)
    CORS_ORIGINS: List[str] = Field(default=["*"])
    LOG_LEVEL: str = Field(default="info")
    
    # Authentication
    SECRET_KEY: str = Field(default=SECRET_KEY)
    JWT_ALGORITHM: str = Field(default=JWT_ALGORITHM)
    JWT_EXPIRATION_DAYS: int = Field(default=30)
    VERIFICATION_TIMEOUT: int = Field(default=300)
    
    # Memory Management
    MEMORY_CLEANUP_INTERVAL: int = Field(default=3600)  # seconds
    MAX_MEMORY_AGE: int = Field(default=86400)  # seconds
    
    # OAuth2 Settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=ACCESS_TOKEN_EXPIRE_MINUTES)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=REFRESH_TOKEN_EXPIRE_DAYS)
    OAUTH2_CLIENT_ID: str = Field(default=OAUTH2_CLIENT_ID)
    OAUTH2_CLIENT_SECRET: str = Field(default=OAUTH2_CLIENT_SECRET)
    OAUTH2_SCOPES: Dict[str, str] = Field(default={
        "chat:read": "Read access to chat data",
        "chat:write": "Write access to chat data",
        "profile:read": "Read access to profile data"
    })
    OAUTH_CLIENTS: Dict[str, Dict] = Field(default=OAUTH_CLIENTS)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = True

@lru_cache()
def get_settings():
    """Get settings singleton instance."""
    try:
        s = Settings()
        print("Settings loaded successfully")
        return s
    except Exception as e:
        print(f"Error loading settings: {e}")
        traceback.print_exc()
        sys.exit(1)

# Initialize settings
try:
    settings = get_settings()
    # For backward compatibility
    config = settings
except Exception as e:
    print(f"Error during settings initialization: {e}")
    traceback.print_exc()
    sys.exit(1)