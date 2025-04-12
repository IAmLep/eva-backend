"""
Configuration module for EVA backend.

This module provides settings and configuration management
using environment variables with Pydantic settings validation.

"""

"""
Version 3 working
"""

import json
import logging
import os
from functools import lru_cache
from typing import List, Optional, Union, Dict, Any

from pydantic import AnyHttpUrl, Field, computed_field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Logger configuration
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    This class defines all configurable settings for the application,
    with default values that can be overridden by environment variables.
    
    Attributes:
        APP_NAME: Name of the application
        APP_VERSION: Current version of the application
        DEBUG: Debug mode flag
        ENVIRONMENT: Running environment (development, testing, production)
        SECRET_KEY: Secret key for JWT token generation
        ALGORITHM: Algorithm for JWT token generation
        ACCESS_TOKEN_EXPIRE_MINUTES: Expiration time for access tokens
        CORS_ORIGINS: List of allowed CORS origins
        SERVICE_URL: Service URL for ID token validation
        DATABASE_URL: Database connection URL
        GOOGLE_CLOUD_PROJECT: Google Cloud project ID
        FIRESTORE_EMULATOR_HOST: Optional Firestore emulator host for local development
        REDIS_HOST: Redis host address
        REDIS_PORT: Redis port
        REDIS_PASSWORD: Redis password
        REDIS_SSL: Whether to use SSL for Redis connection
        GEMINI_API_KEY: Gemini API key
        RATE_LIMIT_PER_MINUTE: Rate limit for API requests per minute
        RATE_LIMIT_PER_DAY: Rate limit for API requests per day
        MEMORY_MAX_TOKENS: Maximum number of tokens to use for memory context
        LOG_LEVEL: Logging level
    """
    
    # General settings
    APP_NAME: str = "EVA Backend"
    APP_VERSION: str = "1.8.6"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="production")
    
    # Security settings
    SECRET_KEY: str = Field(default="", description="Secret key for JWT tokens")
    ALGORITHM: str = Field(default="HS256", description="Algorithm for JWT token generation")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Token expiration time in minutes")
    
    # CORS and API URL settings
    CORS_ORIGINS_STR: Optional[str] = Field(
        default=None,
        description="List of allowed CORS origins as a comma-separated string",
        env="CORS_ORIGINS"  # Map to the same env variable
    )
    SERVICE_URL: str = Field(
        default="https://eva-backend-533306620971.europe-west1.run.app",
        description="Service URL for ID token audience validation"
    )
    
    # Firestore settings
    GOOGLE_CLOUD_PROJECT: Optional[str] = Field(
        default=None, 
        description="Google Cloud project ID"
    )
    FIRESTORE_EMULATOR_HOST: Optional[str] = Field(
        default=None, 
        description="Optional Firestore emulator host for local development"
    )
    
    # API settings
    GEMINI_API_KEY: Optional[str] = Field(
        default=None, 
        description="Gemini API key"
    )
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60, 
        description="Rate limit for API requests per minute"
    )
    RATE_LIMIT_PER_DAY: int = Field(
        default=1000, 
        description="Rate limit for API requests per day"
    )
    
    # Memory settings
    MEMORY_MAX_TOKENS: int = Field(
        default=2000, 
        description="Maximum number of tokens to use for memory context"
    )
    
    # Logging settings
    LOG_LEVEL: str = Field(
        default="INFO", 
        description="Logging level"
    )
    
    @computed_field
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """
        Get parsed CORS origins.
        
        Returns:
            List[str]: List of allowed CORS origins
        """
        if not self.CORS_ORIGINS_STR:
            return []
        
        try:
            # First try to parse as JSON
            return json.loads(self.CORS_ORIGINS_STR)
        except json.JSONDecodeError:
            # If that fails, try to parse as comma-separated string
            return [origin.strip() for origin in self.CORS_ORIGINS_STR.split(",") if origin.strip()]
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """
        Check if environment is production.
        
        Returns:
            bool: True if production environment
        """
        return self.ENVIRONMENT.lower() == "production"
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """
        Check if environment is development.
        
        Returns:
            bool: True if development environment
        """
        return self.ENVIRONMENT.lower() == "development"
    
    @computed_field
    @property
    def is_testing(self) -> bool:
        """
        Check if environment is testing.
        
        Returns:
            bool: True if testing environment
        """
        return self.ENVIRONMENT.lower() == "testing"
    
    @computed_field
    @property
    def firestore_settings(self) -> Dict[str, Any]:
        """
        Get Firestore settings.
        
        Returns:
            Dict[str, Any]: Firestore configuration dictionary
        """
        settings = {}
        
        if self.GOOGLE_CLOUD_PROJECT:
            settings["project_id"] = self.GOOGLE_CLOUD_PROJECT
        
        if self.FIRESTORE_EMULATOR_HOST:
            settings["emulator_host"] = self.FIRESTORE_EMULATOR_HOST
        
        return settings
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        validate_default=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to avoid reloading settings on every call.
    
    Returns:
        Settings: Application settings
    """
    try:
        settings = Settings()
        log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
        
        # Setup basic logging config
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        
        # Log settings when in development mode
        if settings.is_development:
            logger.info(f"Loaded settings for {settings.ENVIRONMENT} environment")
            # Log non-sensitive settings
            safe_settings = settings.model_dump(exclude={"SECRET_KEY", "GEMINI_API_KEY"})
            logger.debug(f"Settings: {safe_settings}")
        
        return settings
    except Exception as e:
        # Add robust error handling to help diagnose settings issues
        logger.error(f"Error loading settings: {str(e)}")
        # Provide a fallback settings object with defaults
        return Settings()