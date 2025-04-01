import logging
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from auth import router as auth_router
from auth_router import router as auth_api_router
from config import get_settings
from error_middleware import ErrorHandlerMiddleware
from exceptions import CustomException
from logging_config import configure_logging
from memory_extractor import router as memory_router
from secrets_router import router as secrets_router
from websocket_manager import router as websocket_router


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
        allow_origins=settings.CORS_ORIGINS,
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
    
    # Register routers
    app.include_router(auth_router.router, prefix="/auth", tags=["Authentication"])
    app.include_router(auth_api_router, prefix="/api/auth", tags=["Auth API"])
    app.include_router(secrets_router, prefix="/api/secrets", tags=["Secrets"])
    app.include_router(memory_router, prefix="/api/memory", tags=["Memory"])
    app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint for Google Cloud Run."""
        return {"status": "healthy"}
    
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=get_settings().DEBUG,
        log_level="info",
    )