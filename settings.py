"""
Settings module for EVA backend.

This module provides configuration utilities beyond what's in config.py,
including environment-specific settings management and override capabilities.

Last updated: 2025-04-01 11:06:46
Version: v1.8.6
Created by: IAmLep
"""

import json
import logging
import os
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, Generic

from pydantic import AnyHttpUrl, BaseModel, EmailStr, Field, validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import get_settings as get_base_settings

# Logger configuration
logger = logging.getLogger(__name__)

# Type variables for generic settings
T = TypeVar('T')


class EnvironmentType(str, Enum):
    """
    Environment type enumeration.
    
    Defines possible deployment environments for the application.
    
    Attributes:
        DEVELOPMENT: Local development environment
        TESTING: Automated testing environment
        STAGING: Pre-production environment
        PRODUCTION: Production environment
    """
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class FeatureFlag(str, Enum):
    """
    Feature flag enumeration.
    
    Defines possible feature flags for enabling/disabling features.
    
    Attributes:
        VOICE_INTERFACE: Voice conversation interface
        ADVANCED_SEARCH: Enhanced semantic search
        IMPROVED_MEMORY: Enhanced memory management
        DATA_VISUALIZATION: Data visualization features
        OFFLINE_MODE: Full offline capability
    """
    VOICE_INTERFACE = "voice_interface"
    ADVANCED_SEARCH = "advanced_search"
    IMPROVED_MEMORY = "improved_memory"
    DATA_VISUALIZATION = "data_visualization"
    OFFLINE_MODE = "offline_mode"


class ServiceConfig(BaseModel):
    """
    Service configuration model.
    
    Contains configuration for an external service integration.
    
    Attributes:
        enabled: Whether the service is enabled
        api_key: Optional API key for the service
        endpoint: Optional service endpoint URL
        timeout_seconds: Connection timeout in seconds
        retry_attempts: Number of retry attempts
        options: Additional service-specific options
    """
    enabled: bool = False
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    options: Dict[str, Any] = Field(default_factory=dict)


class LLMConfig(ServiceConfig):
    """
    Language model configuration.
    
    Extends ServiceConfig with LLM-specific settings.
    
    Attributes:
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty parameter
        presence_penalty: Presence penalty parameter
    """
    model: str = "gemini-1.5-pro"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class OverridableField(Generic[T]):
    """
    Overridable setting field.
    
    Allows a setting to be overridden at runtime for testing or debugging.
    
    Attributes:
        name: Field name
        default_value: Default field value
        current_value: Current field value
        is_overridden: Whether the field is currently overridden
    """
    
    def __init__(self, name: str, default_value: T):
        """
        Initialize overridable field.
        
        Args:
            name: Field name
            default_value: Default field value
        """
        self.name = name
        self.default_value = default_value
        self.current_value = default_value
        self.is_overridden = False
    
    def get(self) -> T:
        """
        Get current field value.
        
        Returns:
            T: Current field value
        """
        return self.current_value
    
    def override(self, value: T) -> None:
        """
        Override field value.
        
        Args:
            value: New field value
        """
        self.current_value = value
        self.is_overridden = True
        logger.info(f"Setting '{self.name}' overridden with value: {value}")
    
    def reset(self) -> None:
        """Reset field to default value."""
        if self.is_overridden:
            self.current_value = self.default_value
            self.is_overridden = False
            logger.info(f"Setting '{self.name}' reset to default: {self.default_value}")


class ExtendedSettings(BaseModel):
    """
    Extended settings model beyond base config.
    
    This model provides additional settings not covered in the base config,
    as well as environment-specific overrides and runtime configuration.
    
    Attributes:
        environment: Application environment
        debug_mode: Whether debug mode is enabled
        log_level: Logging level
        feature_flags: Enabled feature flags
        maintenance_mode: Whether maintenance mode is enabled
        gemini_config: Gemini API configuration
        firebase_config: Firebase configuration
        storage_config: Storage configuration
        cors_allowed_origins: Allowed CORS origins
        cors_allowed_methods: Allowed CORS methods
        rate_limit_config: Rate limiting configuration
    """
    environment: EnvironmentType = Field(
        default=EnvironmentType.PRODUCTION,
        description="Application environment"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    feature_flags: Set[FeatureFlag] = Field(
        default_factory=set,
        description="Enabled feature flags"
    )
    maintenance_mode: bool = Field(
        default=False,
        description="Enable maintenance mode"
    )
    gemini_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Gemini API configuration"
    )
    firebase_config: ServiceConfig = Field(
        default_factory=ServiceConfig,
        description="Firebase configuration"
    )
    storage_config: ServiceConfig = Field(
        default_factory=ServiceConfig,
        description="Storage configuration"
    )
    cors_allowed_origins: List[str] = Field(
        default_factory=list,
        description="Allowed CORS origins"
    )
    cors_allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods"
    )
    rate_limit_config: Dict[str, int] = Field(
        default_factory=lambda: {
            "requests_per_minute": 60,
            "requests_per_day": 1000,
            "tokens_per_minute": 10000,
            "tokens_per_day": 50000
        },
        description="Rate limiting configuration"
    )
    
    # Runtime overridable fields
    _overrides: Dict[str, OverridableField] = {}
    
    def __init__(self, **data):
        """Initialize settings with environment-specific values."""
        super().__init__(**data)
        
        # Initialize overridable fields
        self._overrides = {
            "debug_mode": OverridableField("debug_mode", self.debug_mode),
            "maintenance_mode": OverridableField("maintenance_mode", self.maintenance_mode),
            "log_level": OverridableField("log_level", self.log_level)
        }
        
        # Load environment-specific settings
        self._load_environment_settings()
    
    def _load_environment_settings(self) -> None:
        """Load environment-specific settings from files or environment."""
        try:
            # Get environment
            env = self.environment.value
            
            # Try to load from environment-specific file
            settings_dir = Path(os.environ.get("SETTINGS_DIR", "./settings"))
            env_file = settings_dir / f"{env}_settings.json"
            
            if env_file.exists():
                with open(env_file, "r") as f:
                    env_settings = json.load(f)
                    
                    # Update settings with environment-specific values
                    for key, value in env_settings.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                            
                    logger.info(f"Loaded environment settings from {env_file}")
        except Exception as e:
            logger.warning(f"Failed to load environment settings: {str(e)}")
    
    def is_feature_enabled(self, feature: Union[str, FeatureFlag]) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            feature: Feature flag to check
            
        Returns:
            bool: True if feature is enabled, False otherwise
        """
        if isinstance(feature, str):
            try:
                feature = FeatureFlag(feature)
            except ValueError:
                return False
                
        return feature in self.feature_flags
    
    def enable_feature(self, feature: Union[str, FeatureFlag]) -> bool:
        """
        Enable a feature flag.
        
        Args:
            feature: Feature flag to enable
            
        Returns:
            bool: True if feature was enabled, False if already enabled
        """
        if isinstance(feature, str):
            try:
                feature = FeatureFlag(feature)
            except ValueError:
                return False
        
        if feature in self.feature_flags:
            return False
            
        self.feature_flags.add(feature)
        logger.info(f"Feature enabled: {feature.value}")
        return True
    
    def disable_feature(self, feature: Union[str, FeatureFlag]) -> bool:
        """
        Disable a feature flag.
        
        Args:
            feature: Feature flag to disable
            
        Returns:
            bool: True if feature was disabled, False if already disabled
        """
        if isinstance(feature, str):
            try:
                feature = FeatureFlag(feature)
            except ValueError:
                return False
        
        if feature not in self.feature_flags:
            return False
            
        self.feature_flags.remove(feature)
        logger.info(f"Feature disabled: {feature.value}")
        return True
    
    def override_setting(self, name: str, value: Any) -> bool:
        """
        Override a setting value at runtime.
        
        Args:
            name: Setting name
            value: New setting value
            
        Returns:
            bool: True if setting was overridden, False if setting not found
        """
        if name in self._overrides:
            self._overrides[name].override(value)
            
            # Update instance attribute if it exists
            if hasattr(self, name):
                setattr(self, name, value)
                
            return True
        return False
    
    def reset_setting(self, name: str) -> bool:
        """
        Reset a setting to its default value.
        
        Args:
            name: Setting name
            
        Returns:
            bool: True if setting was reset, False if setting not found
        """
        if name in self._overrides:
            field = self._overrides[name]
            field.reset()
            
            # Update instance attribute if it exists
            if hasattr(self, name):
                setattr(self, name, field.default_value)
                
            return True
        return False
    
    def get_setting(self, name: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            name: Setting name
            default: Default value if setting not found
            
        Returns:
            Any: Setting value or default
        """
        # Check overrides first
        if name in self._overrides:
            return self._overrides[name].get()
            
        # Then check instance attributes
        if hasattr(self, name):
            return getattr(self, name)
            
        return default
    
    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as a dictionary.
        
        Returns:
            Dict[str, Any]: All settings
        """
        # Start with model fields
        settings_dict = self.model_dump(exclude={"_overrides"})
        
        # Add override status
        settings_dict["overridden_settings"] = {
            name: field.is_overridden
            for name, field in self._overrides.items()
        }
        
        return settings_dict
    
    def is_debug_mode(self) -> bool:
        """
        Check if debug mode is enabled.
        
        Returns:
            bool: True if debug mode is enabled
        """
        return self._overrides["debug_mode"].get()
    
    def is_maintenance_mode(self) -> bool:
        """
        Check if maintenance mode is enabled.
        
        Returns:
            bool: True if maintenance mode is enabled
        """
        return self._overrides["maintenance_mode"].get()
    
    def get_log_level(self) -> str:
        """
        Get current log level.
        
        Returns:
            str: Current log level
        """
        return self._overrides["log_level"].get()
    
    def get_environment_name(self) -> str:
        """
        Get current environment name.
        
        Returns:
            str: Current environment name
        """
        return self.environment.value
    
    def is_production(self) -> bool:
        """
        Check if environment is production.
        
        Returns:
            bool: True if production environment
        """
        return self.environment == EnvironmentType.PRODUCTION
    
    def is_development(self) -> bool:
        """
        Check if environment is development.
        
        Returns:
            bool: True if development environment
        """
        return self.environment == EnvironmentType.DEVELOPMENT


# Singleton instance
_extended_settings: Optional[ExtendedSettings] = None


@lru_cache()
def get_settings() -> ExtendedSettings:
    """
    Get extended settings singleton.
    
    This function combines basic settings with extended settings,
    using environment variables, configuration files, and defaults.
    
    Returns:
        ExtendedSettings: Extended settings instance
    """
    global _extended_settings
    
    if _extended_settings is None:
        # Get base settings
        base_settings = get_base_settings()
        
        # Determine environment
        environment = os.environ.get("ENVIRONMENT", "production").lower()
        env_type = EnvironmentType.PRODUCTION
        
        if environment == "development":
            env_type = EnvironmentType.DEVELOPMENT
        elif environment == "testing":
            env_type = EnvironmentType.TESTING
        elif environment == "staging":
            env_type = EnvironmentType.STAGING
        
        # Create initial settings
        settings_data = {
            "environment": env_type,
            "debug_mode": base_settings.DEBUG,
            "log_level": base_settings.LOG_LEVEL,
        }
        
        # Add feature flags based on environment
        feature_flags = set()
        if env_type in (EnvironmentType.DEVELOPMENT, EnvironmentType.TESTING):
            # Enable all features in development and testing
            feature_flags = {f for f in FeatureFlag}
        elif env_type == EnvironmentType.STAGING:
            # Enable select features in staging
            feature_flags = {
                FeatureFlag.VOICE_INTERFACE,
                FeatureFlag.ADVANCED_SEARCH,
                FeatureFlag.IMPROVED_MEMORY
            }
        else:
            # Enable only stable features in production
            feature_flags = {
                FeatureFlag.VOICE_INTERFACE,
                FeatureFlag.IMPROVED_MEMORY
            }
        
        settings_data["feature_flags"] = feature_flags
        
        # Configure Gemini
        gemini_config = LLMConfig(
            enabled=bool(base_settings.GEMINI_API_KEY),
            api_key=base_settings.GEMINI_API_KEY,
            model="gemini-1.5-pro"
        )
        
        settings_data["gemini_config"] = gemini_config
        
        # Configure Firebase
        firebase_config = ServiceConfig(
            enabled=bool(base_settings.GOOGLE_CLOUD_PROJECT),
            options=base_settings.firestore_settings
        )
        
        settings_data["firebase_config"] = firebase_config
        
        # Configure CORS
        settings_data["cors_allowed_origins"] = [str(origin) for origin in base_settings.CORS_ORIGINS]
        
        # Create extended settings
        _extended_settings = ExtendedSettings(**settings_data)
        
        # Log settings loaded
        env_name = _extended_settings.get_environment_name()
        logger.info(f"Extended settings loaded for {env_name} environment")
        
        if _extended_settings.is_debug_mode():
            logger.debug(f"Debug mode enabled in {env_name} environment")
            
            # Log enabled features
            enabled_features = ", ".join(f.value for f in _extended_settings.feature_flags)
            logger.debug(f"Enabled features: {enabled_features}")
    
    return _extended_settings


def require_feature(feature: Union[str, FeatureFlag]) -> Callable:
    """
    Decorator to require a feature flag for a function.
    
    Args:
        feature: Feature flag to require
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            
            if not settings.is_feature_enabled(feature):
                feature_name = feature.value if isinstance(feature, FeatureFlag) else feature
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Feature '{feature_name}' is not enabled"
                )
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            
            if not settings.is_feature_enabled(feature):
                feature_name = feature.value if isinstance(feature, FeatureFlag) else feature
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Feature '{feature_name}' is not enabled"
                )
            
            return func(*args, **kwargs)
        
        # Determine if function is async or sync
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def get_gemini_config() -> LLMConfig:
    """
    Get Gemini API configuration.
    
    Returns:
        LLMConfig: Gemini API configuration
    """
    settings = get_settings()
    return settings.gemini_config


def get_firebase_config() -> ServiceConfig:
    """
    Get Firebase configuration.
    
    Returns:
        ServiceConfig: Firebase configuration
    """
    settings = get_settings()
    return settings.firebase_config


def is_maintenance_mode() -> bool:
    """
    Check if maintenance mode is enabled.
    
    Returns:
        bool: True if maintenance mode is enabled
    """
    settings = get_settings()
    return settings.is_maintenance_mode()


def has_dynamic_settings_changed() -> bool:
    """
    Check if any dynamic settings have been changed from defaults.
    
    Returns:
        bool: True if any settings have been overridden
    """
    settings = get_settings()
    
    # Check if any settings are overridden
    for name, field in settings._overrides.items():
        if field.is_overridden:
            return True
    
    return False