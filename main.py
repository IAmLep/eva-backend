import logging
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time

# Your existing imports
from database import initialize_database, init_db, engine, Base
from api import router as chat_router, periodic_memory_sync
from api_sync import router as sync_router
from auth_router import router as auth_router
from secrets_router import router as secrets_router
from rate_limiter import setup_limiter
from error_middleware import setup_error_handlers

# Configure structured logging for Google Cloud
class CloudRunFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": super().format(record),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()),
            "component": "eva-backend"
        }
        return json.dumps(log_entry)

# Set up production logging
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(CloudRunFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Eva AI Assistant",
    description="Personal AI Assistant Backend API",
    version="1.0.0"
)

# Configure CORS - restrict to specific origins in production
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://eva-app.example.com",  # Change to your actual domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Device-Token"],
)

# Add security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Add logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"{request.method} {request.url.path} completed in {process_time:.2f}ms with status {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request to {request.url.path} failed: {str(e)}")
        raise

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "online", "service": "Eva AI Assistant"}

# Include routers with proper API versioning
app.include_router(chat_router, prefix="/api")  # Changed from /api/v1
app.include_router(sync_router, prefix="/api/sync", tags=["sync"])  # Changed from /api/v1/sync
app.include_router(auth_router, prefix="/auth", tags=["auth"])  # Changed from /api/v1/auth
app.include_router(secrets_router, prefix="/api/secrets", tags=["secrets"])  # Changed from /api/v1/secrets

# Set up rate limiter
setup_limiter(app)

# Set up error handlers
setup_error_handlers(app)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    initialize_database()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
