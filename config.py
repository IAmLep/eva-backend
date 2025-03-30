"""
Configuration settings for the Eva backend application.
"""
import os
from settings import settings

# JWT Settings
SECRET_KEY = settings.SECRET_KEY
JWT_ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS

# Redis Settings
REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT
REDIS_DB = settings.REDIS_DB
REDIS_PASSWORD = settings.REDIS_PASSWORD

# API Keys (NEVER hardcode these in production)
GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_MODEL = settings.GEMINI_MODEL
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# WebSocket Settings
WS_HEARTBEAT_INTERVAL = 30  # seconds

# Device Token Settings
DEVICE_TOKEN_EXPIRE_DAYS = 365  # 1 year for device tokens

# CORS Settings
CORS_ORIGINS = settings.CORS_ORIGINS

# Cache Settings
CACHE_DURATION = 300  # 5 minutes

# Verification Settings
VERIFICATION_TIMEOUT = settings.VERIFICATION_TIMEOUT

# OAuth2 Settings
OAUTH2_CLIENT_ID = settings.OAUTH2_CLIENT_ID
OAUTH2_CLIENT_SECRET = settings.OAUTH2_CLIENT_SECRET
