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
from datetime import datetime, timezone # Import timezone
from typing import Dict, Any, Callable
from contextvars import ContextVar # Import ContextVar directly

from fastapi import FastAPI, Request, Response, status, Depends # Import Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# --- Configuration and Core Components ---
from config import get_settings
from database import get_db_manager
from memory_manager import get_memory_manager
from context_window import get_context_window
from logging_config import configure_logging, get_logger, request_id_var # Import request_id_var
from error_middleware import ErrorHandlerMiddleware
from security import setup_security
from rate_limiter import rate_limiter_dependency # Import rate limiter

# --- API Routers ---
import api
import api_memory
import api_sync
# import api_tools # Keep commented if no router defined yet
import auth_router
import secrets_router
import websocket_manager

settings = get_settings()
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
# request_id_var defined in logging_config.py
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
        raise e
    finally:
        request_id_var.reset(request_id_token)
    return response

# 3. Security Headers (Applied via setup_security)
setup_security(app)

# 4. CORS
if settings.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-ID", "X-Process-Time"])
    logger.info(f"CORS enabled for origins: {settings.CORS_ORIGINS}")
else:
    logger.warning("CORS allowing all origins.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-ID", "X-Process-Time"])

# --- Event Handlers (Unchanged) ---
@app.on_event("startup")
async def startup_event(): ...
@app.on_event("shutdown")
async def shutdown_event(): ...

# --- API Routers ---
# Apply rate limiting dependency to relevant routers
limiter = Depends(rate_limiter_dependency)

app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"]) # No rate limit on auth usually
app.include_router(api.router, prefix="/api/v1/conversation", tags=["Conversation"], dependencies=[limiter])
app.include_router(api_memory.router, prefix="/api/v1/memory", tags=["Memory"], dependencies=[limiter])
app.include_router(api_sync.router, prefix="/api/v1/sync", tags=["Synchronization"], dependencies=[limiter])
# app.include_router(api_tools.router, prefix="/api/v1/tools", tags=["Tools"], dependencies=[limiter]) # Uncomment when ready
app.include_router(secrets_router.router, prefix="/api/v1/secrets", tags=["Secrets"], dependencies=[limiter])
app.include_router(websocket_manager.router, prefix="/ws", tags=["WebSocket"]) # Rate limiting on WS connect? Maybe not needed here.

# --- Root and Health Check Endpoints (Unchanged) ---
@app.get("/", ...)
async def root(): ...
@app.get("/health", ...)
async def health_check(request: Request): ...

# --- Main Execution (Unchanged) ---
if __name__ == "__main__": ...
