import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List

load_dotenv()

class Settings(BaseSettings):
    """Configuration settings for the application."""

    # Database settings
    DATABASE_URL: str = "sqlite+aiosqlite:///./evacore.db"
    
    # Redis
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: Optional[str] = Field(default="")  # Changed from None to empty string
    REDIS_RETRY_ON_STARTUP: bool = Field(default=True)
    REDIS_MAX_RETRIES: int = Field(default=5)
    REDIS_RETRY_INTERVAL: int = Field(default=2)  # seconds
    
    # Gemini AI
    GEMINI_API_KEY: str = Field(default=os.getenv("GEMINI_API_KEY"))
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
    DEBUG: bool = Field(default=True)
    CORS_ORIGINS: List[str] = Field(default=["*"])
    LOG_LEVEL: str = Field(default="info")
    
    # Authentication
    SECRET_KEY: str = Field(default="your-generated-secure-key-here")
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
        default=os.getenv("OAUTH2_CLIENT_SECRET", os.urandom(24).hex())
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = True

settings = Settings()

__all__ = ['settings']

# Add this to test the settings
if __name__ == "__main__":
    print("Settings loaded successfully:")
    print(f"Host: {settings.HOST}")
    print(f"Port: {settings.PORT}")
    print(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
