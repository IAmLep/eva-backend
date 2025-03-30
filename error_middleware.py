from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from exceptions import BaseEvaException  # Changed from .exceptions to absolute import
import logging

logger = logging.getLogger(__name__)

def setup_error_handlers(app: FastAPI):
    @app.exception_handler(BaseEvaException)
    async def eva_exception_handler(request: Request, exc: BaseEvaException):
        logger.error(f"Eva exception: {exc.code} - {exc.message}", 
                     extra={"details": exc.details})
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details
            }
        )
        
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred",
            }
        )