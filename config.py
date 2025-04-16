"""
Configuration handling for EVA backend.

This module configures all system settings and environment variables.
"""

import os
from typing import Optional, Dict, Any, List

# Updated imports for Pydantic v2.x
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Import for environment file loading
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """System settings loaded from environment variables."""
    
    # Environment settings
    ENVIRONMENT: str = "development"  # Options: development, staging, production
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    LOG_LEVEL: str = "INFO"
    APP_NAME: str = "EVA Backend"
    APP_VERSION: str = "2.0.0"
    
    # API configuration
    API_VERSION: str = "2.0.0"
    API_TITLE: str = "EVA Backend API"
    API_DESCRIPTION: str = "Enhanced Virtual Assistant Backend API"
    
    # Authentication settings
    SECRET_KEY: str = Field(..., min_length=32)
    TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    ALGORITHM: str = "HS256"  # Add this for JWT token generation
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # Add this for token expiration
    
    # Database configuration
    DB_PROVIDER: str = "firebase"
    FIREBASE_CREDENTIALS_PATH: str = "firebase-credentials.json"
    
    # Service URL for authentication
    SERVICE_URL: Optional[str] = None
    
    # Memory system settings
    MEMORY_REFRESH_BATCH_SIZE: int = 5
    MEMORY_MAX_CORE_MEMORIES: int = 50
    MEMORY_MAX_EVENT_MEMORIES: int = 10
    MEMORY_IMPORTANCE_THRESHOLD: int = 5
    
    # LLM Service settings
    LLM_PROVIDER: str = "gemini"
    LLM_API_KEY: Optional[str] = None
    LLM_MAX_TOKENS: int = 8192
    LLM_TEMPERATURE: float = 0.7
    
    # Context window settings
    CONTEXT_MAX_TOKENS: int = 16000
    CONTEXT_MAX_MESSAGES: int = 20
    CONTEXT_SUMMARY_TRIGGER: int = 15
    
    # Feature flags
    FEATURES: Dict[str, bool] = {
        "memory_system": True,
        "conversation_analysis": True,
        "knowledge_integration": True,
        "real_time_responses": True
    }
    
    @property
    def is_production(self) -> bool:
        """
        Check if environment is production
        
        Returns:
            bool: True if environment is production, False otherwise
        """
        return self.ENVIRONMENT.lower() == "production"
    
    @field_validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True
        
# Singleton instance for settings
_settings = None

def get_settings() -> Settings:
    """
    Return a singleton Settings instance.
    
    Returns:
        Settings: The application settings
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings