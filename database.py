import logging
from typing import Generator
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import os
import time

# Uncomment if settings is actually needed
# from settings import settings

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

# Message model definition - defined early to avoid import issues
class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    text = Column(Text)
    response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Message {self.id}>"

# User model definition
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username}>"

# Device model definition
class Device(Base):
    __tablename__ = "devices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_name = Column(String(50), nullable=False)
    device_id = Column(String(100), unique=True, nullable=False)
    last_sync = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Device {self.device_name}>"

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
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")
    summary = relationship("ConversationSummary", uselist=False, back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Conversation {self.id}: {self.title}>"

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

    __table_args__ = (
        Index('idx_chat_message_conversation_id', 'conversation_id'),
    )

    def __repr__(self):
        return f"<ChatMessage {self.id}: {self.role}>"

# ConversationSummary model definition
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), unique=True)
    summary = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="summary")

    def __repr__(self):
        return f"<ConversationSummary {self.id}>"

# Database connection setup
# Get database path from environment or use default
DB_PATH = os.environ.get("DB_PATH", "/mnt/eva-memory/eva.db")
DB_URL = f"sqlite:///{DB_PATH}"

# Add connection retries for mounted storage
def get_engine():
    """
    Creates and returns a SQLAlchemy engine with retry logic for Cloud Run cold starts.
    """
    retries = 5
    for attempt in range(retries):
        try:
            # Check if database directory exists
            db_dir = os.path.dirname(DB_PATH)
            if db_dir and not os.path.exists(db_dir):
                logger.info(f"Creating database directory: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)
                
            engine = create_engine(
                DB_URL, 
                connect_args={"check_same_thread": False},
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=5
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection established successfully")
            return engine
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Database connection attempt {attempt+1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Database connection failed after {retries} attempts: {e}")
                raise

# Create engine once
try:
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    # Provide a fallback for imports to work
    engine = None
    SessionLocal = None

def get_db() -> Generator[Session, None, None]:
    """
    Dependency function that provides a database session and ensures it's closed after use.
    """
    if SessionLocal is None:
        logger.error("Cannot get database session: SessionLocal is None")
        raise RuntimeError("Database connection not established")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_database():
    """
    Create all database tables if they don't exist.
    """
    if engine is None:
        logger.error("Cannot initialize database: engine is None")
        raise RuntimeError("Database engine not established")
    
    try:
        # Explicitly create tables in the correct order to avoid dependency issues
        tables = [
            User.__table__,
            Device.__table__,
            Message.__table__,
            Conversation.__table__,
            ChatMessage.__table__,
            ConversationSummary.__table__
        ]
        
        Base.metadata.create_all(bind=engine, tables=tables)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def check_database_health() -> bool:
    """
    Checks if the database is accessible and operational.
    """
    if engine is None:
        logger.error("Cannot check database health: engine is None")
        return False
    
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

# Ensure all models are exported at the top level for proper importing
__all__ = [
    'User', 'Message', 'Device', 'Conversation', 
    'ChatMessage', 'ConversationSummary',
    'get_db', 'initialize_database', 'check_database_health'
]