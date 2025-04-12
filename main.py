"""
Main application module for EVA backend.

This module sets up the FastAPI application with middleware, routes,
and configuration.


Version 3 working
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from error_middleware import ErrorHandlerMiddleware
from exceptions import CustomException
from logging_config import configure_logging

# Import routers with error handling
try:
    from auth import router as auth_router
except ImportError as e:
    logging.warning(f"Unable to import auth router: {str(e)}")
    auth_router = None

try:
    from auth_router import router as auth_api_router
except ImportError as e:
    logging.warning(f"Unable to import auth_api router: {str(e)}")
    auth_api_router = None

try:
    from memory_extractor import router as memory_router
except ImportError as e:
    logging.warning(f"Unable to import memory router: {str(e)}")
    memory_router = None

try:
    from secrets_router import router as secrets_router
except ImportError as e:
    logging.warning(f"Unable to import secrets router: {str(e)}")
    secrets_router = None

try:
    from websocket_manager import router as websocket_router
except ImportError as e:
    logging.warning(f"Unable to import websocket router: {str(e)}")
    websocket_router = None

# Add import for the API router from api.py
try:
    from api import router as api_router
except ImportError as e:
    logging.warning(f"Unable to import API router: {str(e)}")
    api_router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # Startup operations
    configure_logging()
    logging.info("Application startup complete")
    
    yield
    
    # Shutdown operations
    logging.info("Application shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        description="EVA Backend API",
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS or ["*"],  # Fallback to allow all if None
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom error handling
    app.add_middleware(ErrorHandlerMiddleware)
    
    # Register exception handlers
    @app.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code},
        )
    
    # Register routers with error handling
    if auth_router:
        app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    else:
        logging.warning("Auth router not loaded")
    
    if auth_api_router:
        app.include_router(auth_api_router, prefix="/api/auth", tags=["Auth API"])
    else:
        logging.warning("Auth API router not loaded")
    
    if secrets_router:
        app.include_router(secrets_router, prefix="/api/secrets", tags=["Secrets"])
    else:
        logging.warning("Secrets router not loaded")
    
    if memory_router:
        app.include_router(memory_router, prefix="/api/memory", tags=["Memory"])
    else:
        logging.warning("Memory router not loaded")
    
    if websocket_router:
        app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
    else:
        logging.warning("WebSocket router not loaded")
    
    # Add the API router from api.py
    if api_router:
        app.include_router(api_router, prefix="/api", tags=["API"])
        logging.info("API router loaded successfully")
    else:
        logging.warning("API router not loaded")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint for Google Cloud Run."""
        return {"status": "healthy"}
    
    return app


# Use a try-except block for the app creation to provide a clear error message
try:
    app = create_app()
except Exception as e:
    logging.error(f"Error creating app: {str(e)}")
    # Exit with an error code if app creation fails
    sys.exit(1)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=get_settings().DEBUG,
        log_level="info",
    )