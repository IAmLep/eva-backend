from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Your existing imports
from database import initialize_database, init_db, engine, Base
from api import router as chat_router, periodic_memory_sync

# New import for sync
from api_sync import router as sync_router

# Create FastAPI app
app = FastAPI(
    title="Eva AI API",
    description="API for Eva AI conversational assistant",
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

# Include routers
app.include_router(chat_router)
app.include_router(sync_router, prefix="/api/sync", tags=["sync"])

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    initialize_database()
    # Start periodic tasks if needed
    # await periodic_memory_sync()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)