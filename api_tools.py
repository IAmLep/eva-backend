import time
import logging
import asyncio
import json
from typing import List

import redis.asyncio as redis
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from config import config
from database import get_db
from models import Note
from schemas import NoteCreate, NoteResponse

router = APIRouter()
logger = logging.getLogger(__name__)

redis_client = redis.Redis(
    host=config.REDIS_HOST, port=config.REDIS_PORT, db=0, decode_responses=True
)

# ----- Weather Endpoints -----
async def get_new_weather():
    """Fetches weather data (currently mock data)."""
    try:
        weather_data = {
            "forecast": "Sunny",
            "temperature": "14°C",
            "updated": "just now",
            "message": "The weather is sunny at 14°C as of just now."
        }
        return weather_data
    except Exception as e:
        logger.error("Error fetching weather data: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Weather data not available.") from e

@router.get("/tools/weather")
async def get_weather():
    """Retrieve weather data, using cache if available."""
    weather_key = "weather_data"
    try:
        cached_weather = await redis_client.get(weather_key)
        if cached_weather:
            logger.debug("Returning weather data from cache.")
            return json.loads(cached_weather)
        else:
            logger.debug("Fetching new weather data.")
            new_weather = await get_new_weather()
            await redis_client.setex(weather_key, config.CACHE_DURATION, json.dumps(new_weather))
            return new_weather
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redis connection error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e

# ----- News Endpoints -----
async def get_new_news():
    """Fetches news data (currently mock data)."""
    try:
        news_data = {
            "headlines": [
                "AI breakthrough announced",
                "Market updates",
                "Local sports team wins"
            ],
            "message": "Here are the latest headlines."
        }
        return news_data
    except Exception as e:
        logger.error("Error fetching news data: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="News data not available.") from e

@router.get("/tools/news")
async def get_news():
    """Retrieve news data, using cache if available."""
    news_key = "news_data"
    try:
        cached_news = await redis_client.get(news_key)
        if cached_news:
            logger.debug("Returning news data from cache.")
            return json.loads(cached_news)
        else:
            logger.debug("Fetching new news data.")
            new_news = await get_new_news()
            await redis_client.setex(news_key, config.CACHE_DURATION, json.dumps(new_news))
            return new_news
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redis connection error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e

# ----- Notes Endpoints -----
@router.get("/tools/notes", response_model=List[NoteResponse])
async def get_notes(db: AsyncSession = Depends(get_db)):
    """Retrieve all notes from the database."""
    try:
        result = await db.execute(select(Note))
        notes = (await result.scalars().fetchall())
        return notes
    except SQLAlchemyError as e:
        logger.exception(f"Error retrieving notes: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error") from e
    except Exception as e:
        logger.exception(f"Error retrieving notes: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@router.post("/tools/notes", response_model=NoteResponse, status_code=201)
async def create_note(note: NoteCreate, db: AsyncSession = Depends(get_db)):
    """Create a new note in the database."""
    try:
        timestamp = int(time.time() * 1000)
        new_note = Note(title=note.title, content=note.content, timestamp=timestamp)
        db.add(new_note)
        await db.commit()
        return new_note
    except SQLAlchemyError as e:
        logger.exception(f"Error creating note: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error") from e
    except Exception as e:
        logger.exception(f"Error creating note: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

@router.delete("/tools/notes/{note_id}", status_code=204)
async def delete_note(note_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a note from the database."""
    try:
        result = await db.execute(select(Note).where(Note.id == note_id))
        note = result.scalars().first()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        await db.delete(note)
        await db.commit()
    except SQLAlchemyError as e:
        logger.exception(f"Error deleting note: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error") from e
    except Exception as e:
        logger.exception(f"Error deleting note: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

async def periodic_tools_update() -> None:
    """Periodically update the tools cache."""
    while True:
        try:
            logger.debug("Running periodic tools update.")
            # Add your tool update logic here if needed.
            await asyncio.sleep(3600)  # Update every hour
        except Exception as e:
            logger.error(f"Error in periodic tools update: {e}")
            await asyncio.sleep(60)  # Wait before retrying
