import os
import logging.config
import asyncio
import uvicorn  # Move this import to the top
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from logging_config import LOGGING_CONFIG
from auth_router import router as auth_router
from api import router as chat_router, periodic_memory_sync
from api_tools import router as tools_router, periodic_tools_update
from settings import settings
from database import initialize_database, init_db, engine, Base
from rate_limiter import setup_limiter
from redis_manager import RedisManager
from config import IS_PRODUCTION

# Configure logging only once
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    try:
        # Initialize database
        await initialize_database()
        logger.info("Database initialized")
        
        # Start background tasks
        asyncio.create_task(periodic_memory_sync())
        asyncio.create_task(periodic_tools_update())
        logger.info("Background tasks started successfully.")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    yield
    logger.info("Shutting down...")

# Add before creating the FastAPI app
def init_db():
    Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EvaCore - Eva Assistant Backend",
    lifespan=lifespan,
    debug=not IS_PRODUCTION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_limiter(app)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.get("/test", tags=["Health Check"])
async def test_route():
    logger.debug("Test route called")
    return {"status": "ok", "message": "Server is running"}

logger.debug("Including auth router...")
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
logger.debug("Including chat router...")
app.include_router(
    chat_router, 
    prefix="/chat", 
    tags=["Chat"]
)
logger.debug("Including tools router...")
app.include_router(tools_router, prefix="/tools", tags=["Tools"])

@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "Eva backend is running"}

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": "Server is healthy"}

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(f"Rate limit exceeded for client: {request.client.host}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please try again later."},
    )

# Define port before the if/else to make it available in all scopes
port = int(os.getenv("PORT", "8080"))

# Call this function during startup
@app.on_event("startup")
async def startup_event():
    # Initialize database tables
    initialize_database()
    # Your other startup code here

if __name__ == '__main__':
    import socket
    import uvicorn
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
    
    if not IS_PRODUCTION:  # Only check port in development
        # Check if port is in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
        except socket.error as e:
            logger.error(f"Port {port} is in use. Error: {e}")
            exit(1)
    
    # Start the server - this runs in local development only
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=not IS_PRODUCTION)
