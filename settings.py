"""
Settings module for EVA backend.

This module provides configuration utilities beyond what's in config.py,
including environment-specific settings management and override capabilities.

Version 3 working
"""

import json
import logging
import os
import sys
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union, Generic

from pydantic import AnyHttpUrl, BaseModel, EmailStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import settings as get_base_settings

# Logger configuration
logger = logging.getLogger(__name__)

# Type variables for generic settings
T = TypeVar('T')


class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class FeatureFlag(str, Enum):
    VOICE_INTERFACE = "voice_interface"
    ADVANCED_SEARCH = "advanced_search"
    IMPROVED_MEMORY = "improved_memory"
    DATA_VISUALIZATION = "data_visualization"
    OFFLINE_MODE = "offline_mode"


class ServiceConfig(BaseModel):
    enabled: bool = False
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    options: Dict[str, Any] = Field(default_factory=dict)


class LLMConfig(ServiceConfig):
    model: str = "gemini-1.5-pro"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class OverridableField(Generic[T]):
    def __init__(self, name: str, default_value: T):
        self.name = name
        self.default_value = default_value
        self.current_value = default_value
        self.is_overridden = False

    def get(self) -> T:
        return self.current_value

    def override(self, value: T) -> None:
        self.current_value = value
        self.is_overridden = True
        logger.info(f"Setting '{self.name}' overridden with value: {value}")

    def reset(self) -> None:
        if self.is_overridden:
            self.current_value = self.default_value
            self.is_overridden = False
            logger.info(f"Setting '{self.name}' reset to default: {self.default_value}")


class ExtendedSettings(BaseModel):
    environment: EnvironmentType = Field(default=EnvironmentType.PRODUCTION)
    debug_mode: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    feature_flags: Set[FeatureFlag] = Field(default_factory=set)
    maintenance_mode: bool = Field(default=False)
    gemini_config: LLMConfig = Field(default_factory=LLMConfig)
    firebase_config: ServiceConfig = Field(default_factory=ServiceConfig)
    storage_config: ServiceConfig = Field(default_factory=ServiceConfig)
    cors_allowed_origins: List[str] = Field(default_factory=list)
    cors_allowed_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    rate_limit_config: Dict[str, int] = Field(default_factory=lambda: {
        "requests_per_minute": 60,
        "requests_per_day": 1000,
        "tokens_per_minute": 10000,
        "tokens_per_day": 50000
    })

    # NEW FIELD: context window max tokens
    CONTEXT_MAX_TOKENS: int = Field(
        default=int(os.environ.get("CONTEXT_MAX_TOKENS", 2048)),
        description="Maximum context tokens for LLM prompts"
    )

    # Runtime overridable fields
    _overrides: Dict[str, OverridableField] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self._overrides = {
            "debug_mode": OverridableField("debug_mode", self.debug_mode),
            "maintenance_mode": OverridableField("maintenance_mode", self.maintenance_mode),
            "log_level": OverridableField("log_level", self.log_level),
        }
        self._load_environment_settings()

    def _load_environment_settings(self) -> None:
        try:
            env = self.environment.value
            settings_dir = Path(os.environ.get("SETTINGS_DIR", "./settings"))
            env_file = settings_dir / f"{env}_settings.json"
            if env_file.exists():
                with open(env_file, "r") as f:
                    env_settings = json.load(f)
                for key, value in env_settings.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                logger.info(f"Loaded environment settings from {env_file}")
        except Exception as e:
            logger.warning(f"Failed to load environment settings: {str(e)}")

    # ... existing methods (is_feature_enabled, enable_feature, etc.) ...


# Singleton instance
_extended_settings: Optional[ExtendedSettings] = None

@lru_cache()
def get_settings() -> ExtendedSettings:
    global _extended_settings
    if _extended_settings is None:
        base = get_base_settings()
        env_str = os.environ.get("ENVIRONMENT", "production").lower()
        env_type = EnvironmentType(env_str) if env_str in EnvironmentType.__members__.values() else EnvironmentType.PRODUCTION

        # Prepare initial dict
        settings_data = {
            "environment": env_type,
            "debug_mode": base.DEBUG,
            "log_level": base.LOG_LEVEL,
            "feature_flags": set(),  # will populate below
            "CONTEXT_MAX_TOKENS": int(os.environ.get("CONTEXT_MAX_TOKENS", 2048)),
        }

        # Feature flags based on env
        if env_type in (EnvironmentType.DEVELOPMENT, EnvironmentType.TESTING):
            settings_data["feature_flags"] = set(FeatureFlag)
        elif env_type == EnvironmentType.STAGING:
            settings_data["feature_flags"] = {
                FeatureFlag.VOICE_INTERFACE,
                FeatureFlag.ADVANCED_SEARCH,
                FeatureFlag.IMPROVED_MEMORY,
            }
        else:
            settings_data["feature_flags"] = {
                FeatureFlag.VOICE_INTERFACE,
                FeatureFlag.IMPROVED_MEMORY,
            }

        # Gemini config
        settings_data["gemini_config"] = LLMConfig(
            enabled=bool(base.GEMINI_API_KEY),
            api_key=base.GEMINI_API_KEY,
            model=base.GEMINI_MODEL or "gemini-1.5-pro"
        )

        # Firebase config
        settings_data["firebase_config"] = ServiceConfig(
            enabled=bool(base.GOOGLE_CLOUD_PROJECT),
            options=base.firestore_settings
        )

        # CORS origins
        settings_data["cors_allowed_origins"] = [str(o) for o in base.CORS_ORIGINS]

        _extended_settings = ExtendedSettings(**settings_data)
        logger.info(f"Extended settings loaded for {env_type.value} environment")

    return _extended_settings