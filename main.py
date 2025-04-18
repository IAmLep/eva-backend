"""
Main application entry point for EVA backend.

Initializes the FastAPI application, configures middleware, logging,
exception handlers, and registers all API routers.
"""

import asyncio
import logging
import os
import sys
import time
import uuid # Import uuid
from datetime import datetime
from typing import Dict, Any, Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# --- Configuration and Core Components ---
from config import get_settings
from database import get_db_manager
from memory_manager import get_memory_manager
from context_window import get_context_window
from logging_config import configure_logging, get_logger # Use our logging config
from error_middleware import ErrorHandlerMiddleware # Use our error middleware
from security import SecurityMiddleware, setup_security # Import security setup

# --- API Routers ---
import api
import api_memory
import api_sync
import api_tools # Assuming this exists and has a router
import auth_router # Renamed from auth to avoid conflict
import secrets_router
import websocket_manager

# --- Initialize Settings ---
# This should be one of the first things to run
settings = get_settings()

# --- Configure Logging ---
# Do this *after* getting settings, as log level depends on settings
configure_logging()
logger = get_logger(__name__) # Get root logger configured by logging_config

# --- Create FastAPI Application ---
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    openapi_url="/api/openapi.json", # Standard prefix for API docs
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

from security import setup_security
setup_security(app)

# --- Middleware Configuration ---

# 1. Error Handling Middleware (should be early)
app.add_middleware(ErrorHandlerMiddleware)

# 2. Request ID and Logging Middleware
@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Callable):
    """Adds request ID, logs request/response, measures process time."""
    request_id = str(uuid.uuid4()) # Use UUID for request ID
    request.state.request_id = request_id

    # Add request_id to context for logging
    from contextvars import ContextVar
    request_id_var: ContextVar[str] = ContextVar('request_id', default='no-request-id')
    request_id_token = request_id_var.set(request_id)

    logger.info(f"Request started: {request.method} {request.url.path} (ID: {request_id})")
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}" # Format time
        logger.info(f"Request finished: {request.method} {request.url.path} "
                    f"Status: {response.status_code} Process Time: {process_time:.4f}s (ID: {request_id})")
    except Exception as e:
        process_time = time.time() - start_time
        logger.exception(f"Request failed: {request.method} {request.url.path} "
                         f"Process Time: {process_time:.4f}s (ID: {request_id})",
                         exc_info=e)
        # Let the ErrorHandlerMiddleware handle the response generation
        raise e # Re-raise the exception
    finally:
        # Reset context var
        request_id_var.reset(request_id_token)

    return response

# 3. Security Middleware (CSRF, etc. - if needed beyond API routes)
# app.add_middleware(SecurityMiddleware) # Add if you have non-API web routes

# 4. CORS Middleware (applied based on settings)
# Ensure settings.CORS_ORIGINS is correctly populated
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"], # Allow all standard methods
        allow_headers=["*"], # Allow all headers
        expose_headers=["X-Request-ID", "X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"] # Expose custom headers
    )
    logger.info(f"CORS enabled for origins: {settings.CORS_ORIGINS}")
else:
    logger.warning("CORS is not configured. Allowing all origins by default.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
    )

# --- Event Handlers ---

@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup."""
    logger.info(f"--- Starting {settings.API_TITLE} v{settings.API_VERSION} ---")
    logger.info(f"Environment: {settings.ENVIRONMENT} | Debug Mode: {settings.DEBUG}")

    # Initialize database connection (already handled by singleton pattern)
    try:
        get_db_manager()
        logger.info("Database manager initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize Database Manager during startup.", exc_info=e)
        # Depending on severity, you might want to prevent startup
        # sys.exit("Critical component (Database) failed to initialize.")

    # Initialize other components (handled by singletons)
    get_memory_manager()
    get_context_window()
    logger.info("Memory Manager and Context Window initialized.")

    # Log feature flags
    enabled_features = [name for name, enabled in settings.FEATURES.items() if enabled]
    logger.info(f"Enabled features: {enabled_features if enabled_features else 'None'}")
    logger.info("--- Application Startup Complete ---")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("--- Application Shutting Down ---")
    # Add any cleanup logic here (e.g., close database connections if not managed automatically)
    logger.info("--- Application Shutdown Complete ---")

# --- API Routers ---
# Prefixing API routes for better organization

# Authentication
app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"])

# Core Conversation API (REST and WebSocket)
app.include_router(api.router, prefix="/api/v1/conversation", tags=["Conversation"]) # REST part
app.include_router(websocket_manager.router, prefix="/ws", tags=["WebSocket"]) # WebSocket part

# Memory Management
app.include_router(api_memory.router, prefix="/api/v1/memory", tags=["Memory"])

# Synchronization
app.include_router(api_sync.router, prefix="/api/v1/sync", tags=["Synchronization"])

# Tools (Function Calling)
# Assuming api_tools.py has a router defined
# app.include_router(api_tools.router, prefix="/api/v1/tools", tags=["Tools"])

# Secrets Management
app.include_router(secrets_router.router, prefix="/api/v1/secrets", tags=["Secrets"])

# --- Root and Health Check Endpoints ---

@app.get("/", tags=["Status"], include_in_schema=False) # Exclude from OpenAPI docs if desired
async def root():
    """Root endpoint providing basic API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "online",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": settings.ENVIRONMENT,
        "documentation": "/api/docs"
    }

@app.get("/health", tags=["Status"])
async def health_check(request: Request):
    """Health check endpoint for monitoring systems."""
    # Basic health check, can be expanded to check DB, LLM connectivity etc.
    db_status = "up"
    try:
        # Simple check, e.g., try getting a non-existent user
        db = get_db_manager()
        await db.get_user("healthcheck_dummy_id")
    except Exception:
        # Don't log error here, just report status
        db_status = "down" # Or "degraded"

    return {
        "status": "healthy" if db_status == "up" else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": request.state.request_id, # Include request ID
        "components": {
            "api": "up",
            "database": db_status,
            # Add checks for other critical components like LLM if needed
        }
    }

# --- Main Execution ---
# The Dockerfile CMD will run this using Uvicorn
if __name__ == "__main__":
    # This block is mainly for local development
    # Use the port from settings (which respects $PORT env var)
    port = settings.PORT
    host = settings.HOST

    logger.info(f"Starting development server at http://{host}:{port}")
    import uvicorn
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=settings.LOG_LEVEL.lower(), # Use log level from settings
        reload=settings.DEBUG, # Enable reload only if DEBUG is True
        # Consider adding --workers 1 for local dev simplicity if needed
    )
