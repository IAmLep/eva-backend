#!/usr/bin/env python3
"""
EVA AI Assistant Backend
Main application module that handles server setup and lifecycle

Author: IAmLep
Created: 2025-03-31
"""

import os
import time
import logging
import json
from datetime import datetime
import secrets
from typing import Any, Dict, List, Optional, Callable, Union

# FastAPI components
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Firebase components
import firebase_admin
from firebase_admin import credentials

# Local modules
from database import initialize_database
from api import router as chat_router
from api_sync import router as sync_router
from auth_router import router as auth_router
from secrets_router import router as secrets_router
from rate_limiter import setup_limiter
from error_middleware import setup_error_handlers
from logging_config import setup_logging

# ========================================================================
# Environment Configuration
# ========================================================================

class EnvConfig:
    """Environment configuration loader"""
    
    @staticmethod
    def get_bool(name: str, default: bool = False) -> bool:
        """Get boolean value from environment"""
        val = os.environ.get(name, "").lower()
        if not val:
            return default
        return val in ("1", "true", "yes", "on", "y")
    
    @staticmethod
    def get_str(name: str, default: str = "") -> str:
        """Get string value from environment"""
        return os.environ.get(name, default)
    
    @staticmethod
    def get_int(name: str, default: int = 0) -> int:
        """Get integer value from environment"""
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def get_list(name: str, default: List[str] = None) -> List[str]:
        """Get list of strings from comma-separated environment variable"""
        if default is None:
            default = []
        val = os.environ.get(name, "")
        if not val:
            return default
        return [item.strip() for item in val.split(",") if item.strip()]

# Load configuration
config = {
    "app_name": "EVA AI Assistant",
    "version": "1.8.6",
    "is_prod": EnvConfig.get_bool("PRODUCTION") or EnvConfig.get_str("ENVIRONMENT") == "production",
    "is_cloud": EnvConfig.get_bool("CLOUD_ENV") or EnvConfig.get_bool("CLOUD_ENVIRONMENT"),
    "port": EnvConfig.get_int("PORT", 8080),
    "log_level": EnvConfig.get_str("LOG_LEVEL", "INFO"),
    "allowed_origins": EnvConfig.get_list("ALLOWED_ORIGINS", ["https://eva-backend-533306620971.europe-west1.run.app"])
}

# ========================================================================
# Logging Setup
# ========================================================================

class EVALogger:
    """Custom logging setup for EVA backend"""
    
    class JSONFormatter(logging.Formatter):
        """Format logs as JSON objects"""
        
        def format(self, record: logging.LogRecord) -> str:
            """Format log record as JSON"""
            # Basic log data
            output = {
                "time": self.formatTime(record),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "line": record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                exc_type = record.exc_info[0].__name__ if record.exc_info[0] else "Unknown"
                exc_msg = str(record.exc_info[1])
                output["error"] = {"type": exc_type, "message": exc_msg}
            
            # Add request ID if available
            if hasattr(record, "request_id"):
                output["request_id"] = record.request_id
                
            return json.dumps(output)
    
    @staticmethod
    def setup() -> logging.Logger:
        """Set up and configure logging"""
        # Set up base logging through the logging_config module
        logger = setup_logging()
        
        # Add custom JSON handler for structured logging
        handler = logging.StreamHandler()
        handler.setFormatter(EVALogger.JSONFormatter())
        
        # Get root logger and reconfigure
        root = logging.getLogger()
        
        # Remove existing handlers to avoid duplication
        for h in root.handlers[:]:
            root.removeHandler(h)
            
        # Add our handler
        root.addHandler(handler)
        
        # Set level from config
        level = getattr(logging, config["log_level"].upper(), logging.INFO)
        root.setLevel(level)
        
        return logger

# Initialize logger
logger = EVALogger.setup()
logger.info(f"Starting EVA backend v{config['version']} in {'production' if config['is_prod'] else 'development'} mode")

# ========================================================================
# Security Components
# ========================================================================

class SecurityHeaders:
    """Manages security headers for HTTP responses"""
    
    def __init__(self, is_production: bool):
        """Initialize with appropriate headers based on environment"""
        self.is_production = is_production
        
        # Base headers used in all environments
        self.base_headers = {
            "X-Content-Type-Options": "nosniff"
        }
        
        # Production-only headers
        self.prod_headers = {
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()"
        }
        
        # Generate Content Security Policy
        self.csp = self._build_csp()
    
    def _build_csp(self) -> str:
        """Build Content Security Policy header value"""
        directives = {
            "default-src": ["'self'"],
            "script-src": ["'self'", "'unsafe-inline'", "https://www.googletagmanager.com", 
                          "https://www.google-analytics.com"],
            "style-src": ["'self'", "'unsafe-inline'"],
            "img-src": ["'self'", "data:", "https://www.google-analytics.com"],
            "font-src": ["'self'"],
            "object-src": ["'none'"],
            "frame-ancestors": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"]
        }
        
        # Convert the directives dictionary to CSP string
        parts = []
        for directive, sources in directives.items():
            parts.append(f"{directive} {' '.join(sources)}")
        
        return "; ".join(parts)
    
    def apply(self, response: Response) -> None:
        """Apply security headers to a response"""
        # Apply base headers
        for name, value in self.base_headers.items():
            response.headers[name] = value
        
        # Apply production headers if in production mode
        if self.is_production:
            for name, value in self.prod_headers.items():
                response.headers[name] = value
            # Apply CSP
            response.headers["Content-Security-Policy"] = self.csp

# Create security headers manager
security = SecurityHeaders(config["is_prod"])

# ========================================================================
# Firebase Management
# ========================================================================

class FirebaseManager:
    """Manages Firebase initialization and interaction"""
    
    @staticmethod
    def check_initialized() -> bool:
        """Check if Firebase is already initialized"""
        try:
            firebase_admin.get_app()
            return True
        except ValueError:
            return False
    
    @staticmethod
    def initialize() -> bool:
        """Initialize Firebase with fallback strategies"""
        # Skip if already initialized
        if FirebaseManager.check_initialized():
            logger.info("Firebase already initialized")
            return True
        
        # Try initializing with different methods
        if FirebaseManager._init_cloud():
            return True
        
        if FirebaseManager._init_credentials_file():
            return True
        
        if FirebaseManager._init_default():
            return True
        
        logger.error("All Firebase initialization methods failed")
        return False
    
    @staticmethod
    def _init_cloud() -> bool:
        """Initialize Firebase using cloud credentials"""
        if not config["is_cloud"]:
            return False
        
        try:
            logger.info("Initializing Firebase with cloud credentials")
            firebase_admin.initialize_app()
            logger.info("Firebase initialized successfully with cloud credentials")
            return True
        except Exception as e:
            logger.warning(f"Cloud credentials initialization failed: {str(e)}")
            return False
    
    @staticmethod
    def _init_credentials_file() -> bool:
        """Initialize Firebase using credentials file"""
        # Try multiple credential file locations
        paths = [
            EnvConfig.get_str("FIREBASE_CREDENTIALS"),
            EnvConfig.get_str("FIREBASE_CREDENTIALS_PATH"),
            EnvConfig.get_str("GOOGLE_APPLICATION_CREDENTIALS")
        ]
        
        for path in paths:
            if not path or not os.path.exists(path):
                continue
                
            try:
                logger.info(f"Initializing Firebase with credentials file: {path}")
                cred = credentials.Certificate(path)
                firebase_admin.initialize_app(cred)
                logger.info(f"Firebase initialized successfully with credentials from {path}")
                return True
            except Exception as e:
                logger.warning(f"Credentials file initialization failed: {str(e)}")
        
        return False
    
    @staticmethod
    def _init_default() -> bool:
        """Initialize Firebase with default configuration as last resort"""
        try:
            logger.info("Initializing Firebase with default configuration")
            firebase_admin.initialize_app()
            logger.info("Firebase initialized successfully with default configuration")
            return True
        except Exception as e:
            logger.error(f"Default Firebase initialization failed: {str(e)}")
            return False

# ========================================================================
# FastAPI Application Setup
# ========================================================================

# Create the FastAPI application
app = FastAPI(
    title=config["app_name"],
    description="Backend API for EVA AI Assistant",
    version=config["version"],
    # Disable OpenAPI docs in production for security
    docs_url=None if config["is_prod"] else "/docs",
    redoc_url=None if config["is_prod"] else "/redoc",
    openapi_url=None if config["is_prod"] else "/openapi.json"
)

# CORS configuration
origins = config["allowed_origins"] if config["is_prod"] else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Device-Token"],
)

# ========================================================================
# Middleware
# ========================================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next: Callable) -> Response:
    """Apply security headers to responses"""
    response = await call_next(request)
    security.apply(response)
    return response

@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next: Callable) -> Response:
    """Track request processing time and log details"""
    # Generate request ID
    request_id = f"r-{int(time.time())}-{secrets.token_hex(4)}"
    
    # Start timer
    start_time = time.time()
    
    # Add request ID to log records
    logger_adapter = logging.LoggerAdapter(logger, {"request_id": request_id})
    logger_adapter.info(f"Request started: {request.method} {request.url.path}")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = round((time.time() - start_time) * 1000, 2)
        
        # Log based on status code
        status = response.status_code
        if status >= 500:
            logger_adapter.error(f"Request failed: {status} in {duration}ms - {request.method} {request.url.path}")
        elif status >= 400:
            logger_adapter.warning(f"Request error: {status} in {duration}ms - {request.method} {request.url.path}")
        else:
            logger_adapter.info(f"Request completed: {status} in {duration}ms - {request.method} {request.url.path}")
        
        # Add request ID to response headers in non-production environments
        if not config["is_prod"]:
            response.headers["X-Request-ID"] = request_id
            
        return response
    except Exception as e:
        # Log exception
        duration = round((time.time() - start_time) * 1000, 2)
        logger_adapter.error(f"Request exception after {duration}ms: {str(e)}", exc_info=True)
        raise

# ========================================================================
# Routes
# ========================================================================

@app.get("/")
async def health_check() -> Dict[str, Any]:
    """Service health check endpoint"""
    return {
        "service": config["app_name"],
        "status": "online",
        "version": config["version"],
        "environment": "production" if config["is_prod"] else "development",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds")
    }

# Include routers from modules
app.include_router(chat_router, prefix="/api")
app.include_router(sync_router, prefix="/api/sync", tags=["sync"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(secrets_router, prefix="/api/secrets", tags=["secrets"])

# Configure error handlers
setup_error_handlers(app)

# Set up rate limiting
setup_limiter(app)

# ========================================================================
# Application Lifecycle Events
# ========================================================================

@app.on_event("startup")
async def on_startup() -> None:
    """Initialize services when the application starts"""
    logger.info("Application startup initiated")
    
    # Initialize database
    try:
        initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        error_msg = f"Database initialization failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)
    
    # Initialize Firebase
    firebase_initialized = FirebaseManager.initialize()
    if not firebase_initialized:
        error_msg = "Firebase initialization failed"
        logger.error(error_msg)
        
        # In production environments, this is a critical failure
        if config["is_prod"] or config["is_cloud"]:
            raise RuntimeError(error_msg)
    
    logger.info("Application startup completed successfully")

@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Clean up resources when the application shuts down"""
    logger.info("Application shutdown initiated")
    
    # Add cleanup code here if needed
    
    logger.info("Application shutdown completed")

# ========================================================================
# Development Server
# ========================================================================

if __name__ == "__main__":
    import uvicorn
    
    reload_enabled = not config["is_prod"]
    port = config["port"]
    
    logger.info(f"Starting development server on port {port}{' with reload enabled' if reload_enabled else ''}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level=config["log_level"].lower(),
        reload=reload_enabled
    )