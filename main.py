"""
Main application entry point for EVA backend.

This file initializes the FastAPI application, configures middleware,
and registers all API routers.

Update your existing main.py file with this enhanced version.

Current Date: 2025-04-13 11:20:47
Current User: IAmLep
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from database import get_db_manager
from memory_manager import get_memory_manager
from context_window import get_context_window

# Import API routers
import api
import api_memory
import websocket_manager
from exceptions import AuthorizationError, DatabaseError, LLMServiceError, NotFoundException, RateLimitError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize application
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enhanced EVA AI Assistant API",
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add a unique request ID to each request."""
    request_id = f"req_{int(time.time() * 1000)}"
    request.state.request_id = request_id
    
    # Add to request headers
    request.headers.__dict__["_list"].append(
        (b"x-request-id", request_id.encode())
    )
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add headers to response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Exception handlers
@app.exception_handler(LLMServiceError)
async def llm_service_exception_handler(request: Request, exc: LLMServiceError):
    """Handle LLM service errors."""
    logger.error(f"LLM service error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)}
    )

@app.exception_handler(RateLimitError)
async def rate_limit_exception_handler(request: Request, exc: RateLimitError):
    """Handle rate limit errors."""
    logger.warning(f"Rate limit exceeded: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)}
    )

@app.exception_handler(AuthorizationError)
async def authorization_exception_handler(request: Request, exc: AuthorizationError):
    """Handle authorization errors."""
    logger.warning(f"Authorization error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": str(exc)}
    )

@app.exception_handler(DatabaseError)
async def database_exception_handler(request: Request, exc: DatabaseError):
    """Handle database errors."""
    logger.error(f"Database error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Database error: {str(exc)}"}
    )

@app.exception_handler(NotFoundException)
async def not_found_exception_handler(request: Request, exc: NotFoundException):
    """Handle not found errors."""
    logger.info(f"Resource not found: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)}
    )

# Initialize components
@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize database connection
    db_manager = get_db_manager()
    logger.info("Database manager initialized")
    
    # Initialize memory manager
    memory_manager = get_memory_manager()
    logger.info("Memory manager initialized")
    
    # Initialize context window
    context_window = get_context_window()
    logger.info("Context window initialized")
    
    # Log feature flags
    for feature, enabled in settings.FEATURES.items():
        status_str = "enabled" if enabled else "disabled"
        logger.info(f"Feature '{feature}' is {status_str}")

# Register API routers
app.include_router(api.router)
app.include_router(api_memory.router)
app.include_router(websocket_manager.router)

# Root endpoint
@app.get("/", tags=["status"])
async def root():
    """
    Root endpoint providing API information.
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT
    }

# Health check endpoint
@app.get("/health", tags=["status"])
async def health_check():
    """
    Health check endpoint for monitoring systems.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "up",
            "database": "up",
            "memory": "up"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Use environment variables or defaults
    host = os.getenv("HOST", settings.HOST)
    port = int(os.getenv("PORT", settings.PORT))
    
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=settings.DEBUG)