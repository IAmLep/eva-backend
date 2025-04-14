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
    
    # API configuration
    API_VERSION: str = "2.0.0"
    API_TITLE: str = "EVA Backend API"
    API_DESCRIPTION: str = "Enhanced Virtual Assistant Backend API"
    DEBUG: bool = False
    
    # Authentication settings
    SECRET_KEY: str = Field(..., min_length=32)
    TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    
    # Database configuration
    DB_PROVIDER: str = "firebase"
    
    # Memory system settings
    MEMORY_REFRESH_BATCH_SIZE: int = 5  # Number of memories to refresh per batch
    MEMORY_MAX_CORE_MEMORIES: int = 50  # Maximum number of core memories to load
    MEMORY_MAX_EVENT_MEMORIES: int = 10  # Maximum number of event memories to load
    MEMORY_IMPORTANCE_THRESHOLD: int = 5  # Minimum importance for memories to be loaded
    
    # LLM Service settings
    LLM_PROVIDER: str = "gemini"
    LLM_API_KEY: Optional[str] = None
    LLM_MAX_TOKENS: int = 8192
    LLM_TEMPERATURE: float = 0.7
    
    # Context window settings
    CONTEXT_MAX_TOKENS: int = 16000
    CONTEXT_MAX_MESSAGES: int = 20
    CONTEXT_SUMMARY_TRIGGER: int = 15  # Number of messages before summarization
    
    # Replace validator decorator with field_validator for Pydantic v2.x
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