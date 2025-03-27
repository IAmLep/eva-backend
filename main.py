import logging
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time

# Your existing imports
from database import initialize_database, init_db, engine, Base
from api import router as chat_router, periodic_memory_sync
from api_sync import router as sync_router

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(chat_router, prefix="/api/v1")
app.include_router(sync_router, prefix="/api/v1/sync", tags=["sync"])

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    initialize_database()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
