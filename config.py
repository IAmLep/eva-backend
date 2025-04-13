"""
Configuration module for EVA backend.

This module provides application configuration settings, including
enhanced support for the memory system and context management.

Update your existing config.py file with this version.

Current Date: 2025-04-13 11:03:01
Current User: IAmLepin
"""

import os
from functools import lru_cache
from typing import Dict, Any, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Application settings.
    
    This class provides configuration settings from environment variables
    with reasonable defaults for development.
    
    Attributes:
        PROJECT_NAME: Name of the application
        VERSION: Application version
        ENVIRONMENT: Current environment (development, staging, production)
        DEBUG: Debug mode flag
        
        # Server settings
        PORT: Port to run the server on
        HOST: Host to bind the server to
        WORKERS: Number of worker processes
        
        # Authentication settings
        SECRET_KEY: Secret key for JWT tokens
        ACCESS_TOKEN_EXPIRE_MINUTES: JWT token expiration time
        REFRESH_TOKEN_EXPIRE_DAYS: Refresh token expiration time
        
        # Database settings
        FIREBASE_CREDENTIALS_PATH: Path to Firebase service account credentials
        
        # Gemini API settings
        GEMINI_API_KEY: API key for Gemini API
        GEMINI_URL: Gemini API URL
        
        # Memory system settings
        MEMORY_DEFAULT_LIMIT: Default number of memories to retrieve
        CONTEXT_MAX_TOKENS: Maximum tokens in context window
        SUMMARIZE_AFTER_TURNS: Number of turns before conversation summarization
        
        # Rate limiting
        RATE_LIMIT_ENABLED: Whether rate limiting is enabled
        FREE_TIER_REQUESTS_PER_MINUTE: Rate limit for free tier
        FREE_TIER_TOKENS_PER_DAY: Token limit for free tier
    """
    # General settings
    PROJECT_NAME: str = "EVA AI Assistant"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Server settings
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    WORKERS: int = 4
    
    # Authentication settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    SECURE_COOKIES: bool = False  # Set to True in production
    
    # Database settings
    FIREBASE_CREDENTIALS_PATH: str = "./firebase-credentials.json"
    
    # Gemini API settings
    GEMINI_API_KEY: Optional[str] = Field(None, env="GEMINI_API_KEY")
    GEMINI_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    
    # Memory system settings
    MEMORY_DEFAULT_LIMIT: int = 50
    CONTEXT_MAX_TOKENS: int = 8000  # Gemini's context window size
    SUMMARIZE_AFTER_TURNS: int = 10
    CORE_MEMORY_IMPORTANCE_THRESHOLD: int = 7  # Min importance for core memories
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    FREE_TIER_REQUESTS_PER_MINUTE: int = 20
    FREE_TIER_TOKENS_PER_DAY: int = 100000
    
    # Memory optimization
    ENTITY_TRACKING_ENABLED: bool = True
    MEMORY_IMPORTANCE_SCORING_ENABLED: bool = True
    AUTO_SUMMARIZATION_ENABLED: bool = True
    
    # Mental health settings (for Stage 3)
    MENTAL_HEALTH_SUPPORT_ENABLED: bool = False
    
    # Feature flags
    FEATURES: Dict[str, bool] = {
        "memory_system": True,
        "context_window": True,
        "streaming_responses": True,
        "function_calling": False,  # Will be enabled in Stage 2
        "mental_health": False,     # Will be enabled in Stage 3
        "habit_tracking": False,    # Will be enabled in Stage 3
    }
    
    class Config:
        """Pydantic model configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @validator("SECRET_KEY", pre=True)
    def validate_secret_key(cls, v: Optional[str]) -> str:
        """
        Validate SECRET_KEY.
        
        Generates a random key for development if not provided.
        """
        if not v:
            import secrets
            return secrets.token_urlsafe(32)
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings.
    
    Uses LRU cache to avoid re-creating settings object for every request.
    
    Returns:
        Settings: Application settings
    """
    return Settings()