"""
Main application entry point for EVA backend.

Initializes the FastAPI application, configures middleware, logging,
exception handlers, and registers all API routers. Applies rate limiting.
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Callable, List # Ensure List is imported
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# --- Configuration and Core Components ---
from config import settings
from database import get_db_manager
from memory_manager import get_memory_manager
from context_window import get_context_window
from logging_config import configure_logging, get_logger, request_id_var
from error_middleware import ErrorHandlerMiddleware
from security import setup_security
from rate_limiter import rate_limiter_dependency

# --- API Routers ---
import api
import api_memory
import api_sync
# import api_tools
import auth_router
import secrets_router
import websocket_manager

# Load settings (CORS_ORIGINS is now a string in the settings object)

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# --- Middleware Configuration ---

# 1. Error Handling (Earliest)
app.add_middleware(ErrorHandlerMiddleware)

# 2. Request ID and Logging
@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Callable):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request_id_token = request_id_var.set(request_id)
    logger.info(f"Request started: {request.method} {request.url.path} (ID: {request_id})")
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        logger.info(f"Request finished: {request.method} {request.url.path} Status: {response.status_code} Time: {process_time:.4f}s (ID: {request_id})")
    except Exception as e:
        process_time = time.time() - start_time
        logger.exception(f"Request failed: {request.method} {request.url.path} Time: {process_time:.4f}s (ID: {request_id})", exc_info=e)
        raise e # Re-raise after logging
    finally:
        request_id_var.reset(request_id_token)
    return response

# 3. Security Headers (Applied via setup_security)
setup_security(app)

# 4. CORS (Manual Parsing of the string from settings)
# Get the raw string from settings
cors_origins_str: str = settings.CORS_ORIGINS
parsed_origins: List[str] = [] # Initialize empty list

if cors_origins_str:
    # Split comma-separated string and strip whitespace
    # Filter out empty strings that might result from trailing commas or just whitespace/commas
    parsed_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

# If parsing resulted in an empty list (e.g., input was "" or just ","), default to ["*"]
if not parsed_origins:
    logger.warning(f"CORS_ORIGINS string '{cors_origins_str}' parsed to empty list. Defaulting to allow all ['*'].")
    parsed_origins = ["*"]

# Log the final list being used for CORS configuration
if parsed_origins == ["*"]:
    logger.warning("CORS configured to allow all origins ['*']. This is generally not recommended for production unless intended.")
else:
    logger.info(f"CORS configured for specific origins: {parsed_origins}")

# Add the CORSMiddleware using the manually parsed list
app.add_middleware(
    CORSMiddleware,
    allow_origins=parsed_origins, # Use the manually parsed list
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS, # Use bool directly from settings
    allow_methods=settings.CORS_ALLOW_METHODS,       # Use list directly from settings
    allow_headers=settings.CORS_ALLOW_HEADERS,       # Use list directly from settings
    expose_headers=["X-Request-ID", "X-Process-Time"] # List headers to expose to the browser
)

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Application startup logic."""
    logger.info("Application startup...")
    # Initialize database connections, etc. here if needed
    # Example: await get_db_manager().connect()

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown logic."""
    logger.info("Application shutdown...")
    # Clean up resources, close connections, etc. here if needed
    # Example: await get_db_manager().disconnect()

# --- API Routers ---
# Apply rate limiting dependency to relevant routers
limiter = Depends(rate_limiter_dependency)

app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(api.router, prefix="/api/v1/conversation", tags=["Conversation"], dependencies=[limiter])
app.include_router(api_memory.router, prefix="/api/v1/memory", tags=["Memory"], dependencies=[limiter])
app.include_router(api_sync.router, prefix="/api/v1/sync", tags=["Synchronization"], dependencies=[limiter])
# app.include_router(api_tools.router, prefix="/api/v1/tools", tags=["Tools"], dependencies=[limiter])
app.include_router(secrets_router.router, prefix="/api/v1/secrets", tags=["Secrets"], dependencies=[limiter])
app.include_router(websocket_manager.router, prefix="/ws", tags=["WebSocket"])

# --- Root and Health Check Endpoints ---
@app.get("/", tags=["General"], summary="Root endpoint")
async def root():
    """Provides a simple welcome message."""
    return {"message": f"Welcome to {settings.PROJECT_NAME} API v{settings.API_VERSION}"}

@app.get("/health", tags=["General"], summary="Health check endpoint", status_code=status.HTTP_200_OK)
async def health_check(request: Request):
    """Returns the status of the application."""
    # In a real app, you might check db connections, etc.
    return {"status": "ok", "request_id": request.state.request_id}

# --- Main Execution ---
# This block allows running the app directly with uvicorn for local development
# Example: python main.py
# Note: Cloud Run uses its own entrypoint mechanism (likely 'gunicorn' or 'uvicorn' directly)
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server locally on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG # Enable reload only in debug mode
    )