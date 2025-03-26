import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import os
import time

from settings import settings

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

# User model definition
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversations = relationship("Conversation", back_populates="user")

# Message model definition
class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    text = Column(Text)
    response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Device model definition
class Device(Base):
    __tablename__ = "devices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_name = Column(String(50), nullable=False)
    device_id = Column(String(100), unique=True, nullable=False)
    last_sync = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

# Conversation model definition
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_synced_device = Column(String(100), nullable=True)
    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation")
    summary = relationship("ConversationSummary", uselist=False, back_populates="conversation", cascade="all, delete-orphan")

# ChatMessage model definition
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    device_id = Column(String(100), nullable=True)  # Which device created this message
    conversation = relationship("Conversation", back_populates="messages")

# ConversationSummary model definition
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), unique=True)
    summary = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="summary")

# Get database path from environment or use default
# Use the correct mount path for Cloud Run
DB_PATH = os.environ.get("DB_PATH", "/mnt/eva-memory/eva.db")
DB_URL = f"sqlite:///{DB_PATH}"

# Add connection retries for mounted storage
def get_engine():
    # Retry logic for Cloud Run cold starts when bucket might not be mounted yet
    retries = 5
    for attempt in range(retries):
        try:
            # Check if database directory exists
            db_dir = os.path.dirname(DB_PATH)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                
            engine = create_engine(
                DB_URL, 
                connect_args={"check_same_thread": False},
                pool_pre_ping=True
            )
            return engine
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Database connection attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(1)  # Wait before retry
            else:
                logger.error(f"Database connection failed after {retries} attempts: {e}")
                raise e

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Synchronous database initialization function
def initialize_database():
    """
    Create all database tables if they don't exist
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
