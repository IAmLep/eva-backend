"""
Configuration settings for the Eva backend application.
"""
import os
from settings import settings

class Config:
    # JWT Settings
    SECRET_KEY = settings.SECRET_KEY
    JWT_ALGORITHM = settings.JWT_ALGORITHM
    ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS
    DEVICE_TOKEN_EXPIRE_DAYS = 365  # 1 year for device tokens

    # Redis Settings
    REDIS_HOST = settings.REDIS_HOST
    REDIS_PORT = settings.REDIS_PORT
    REDIS_DB = settings.REDIS_DB
    REDIS_PASSWORD = settings.REDIS_PASSWORD

    # API Keys
    GEMINI_API_KEY = settings.GEMINI_API_KEY
    GEMINI_MODEL = settings.GEMINI_MODEL

    # WebSocket Settings
    WS_HEARTBEAT_INTERVAL = 30  # seconds

    # CORS Settings
    CORS_ORIGINS = settings.CORS_ORIGINS

    # Cache Settings
    CACHE_DURATION = 300  # 5 minutes

    # Verification Settings
    VERIFICATION_TIMEOUT = settings.VERIFICATION_TIMEOUT

    # OAuth2 Settings
    OAUTH2_CLIENT_ID = settings.OAUTH2_CLIENT_ID
    OAUTH2_CLIENT_SECRET = settings.OAUTH2_CLIENT_SECRET

# Create a single instance of the config
config = Config()

# For backwards compatibility, expose all config variables at the module level
SECRET_KEY = config.SECRET_KEY
JWT_ALGORITHM = config.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = config.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = config.REFRESH_TOKEN_EXPIRE_DAYS
DEVICE_TOKEN_EXPIRE_DAYS = config.DEVICE_TOKEN_EXPIRE_DAYS
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_DB = config.REDIS_DB
REDIS_PASSWORD = config.REDIS_PASSWORD
GEMINI_API_KEY = config.GEMINI_API_KEY
GEMINI_MODEL = config.GEMINI_MODEL
WS_HEARTBEAT_INTERVAL = config.WS_HEARTBEAT_INTERVAL
CORS_ORIGINS = config.CORS_ORIGINS
CACHE_DURATION = config.CACHE_DURATION
VERIFICATION_TIMEOUT = config.VERIFICATION_TIMEOUT
OAUTH2_CLIENT_ID = config.OAUTH2_CLIENT_ID
OAUTH2_CLIENT_SECRET = config.OAUTH2_CLIENT_SECRET