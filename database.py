import logging
from typing import Generator, Optional, List, Dict, Any
from sqlalchemy import create_engine, Index, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import os
import time
import warnings
import json

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

# Sync mixin that can be added to all models requiring sync
class SyncMixin:
    last_modified = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sync_status = Column(String(20), default="pending_sync")  # pending_sync, synced
    device_id = Column(String(100), nullable=True)  # Device that last modified this record

# User model definition
class User(Base, SyncMixin):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username}>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "sync_status": self.sync_status,
            "device_id": self.device_id
        }

# Message model definition
class Message(Base, SyncMixin):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    text = Column(Text)
    response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Message {self.id}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "device_id": self.device_id,
            "text": self.text,
            "response": self.response,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "sync_status": self.sync_status
        }

# Device model definition
class Device(Base, SyncMixin):
    __tablename__ = "devices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_name = Column(String(50), nullable=False)
    device_id = Column(String(100), unique=True, nullable=False)
    last_sync = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Device {self.device_name}>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "device_name": self.device_name,
            "device_id": self.device_id,
            "last_sync": self.last_sync.isoformat(),
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "sync_status": self.sync_status
        }

# Conversation model definition
class Conversation(Base, SyncMixin):
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
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_synced_device": self.last_synced_device,
            "last_modified": self.last_modified.isoformat(),
            "sync_status": self.sync_status,
            "device_id": self.device_id
        }

# ChatMessage model definition
class ChatMessage(Base, SyncMixin):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index('idx_chat_message_conversation_id', 'conversation_id'),
    )

    def __repr__(self):
        return f"<ChatMessage {self.id}: {self.role}>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "device_id": self.device_id,
            "last_modified": self.last_modified.isoformat(),
            "sync_status": self.sync_status
        }

# ConversationSummary model definition
class ConversationSummary(Base, SyncMixin):
    __tablename__ = "conversation_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), unique=True)
    summary = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="summary")

    def __repr__(self):
        return f"<ConversationSummary {self.id}>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "summary": self.summary,
            "updated_at": self.updated_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "sync_status": self.sync_status,
            "device_id": self.device_id
        }

# In-memory sync record storage for the server
class SyncRecord(Base):
    """Temporary storage for sync data on the server"""
    __tablename__ = "sync_records"
    
    id = Column(String, primary_key=True)  # Record UUID/ID
    table_name = Column(String(50), nullable=False)  # Which table this belongs to
    record_data = Column(Text, nullable=False)  # JSON data of the record
    device_id = Column(String(100), nullable=False)  # Device that sent this record
    timestamp = Column(DateTime, default=datetime.utcnow)  # When this record was received
    processed = Column(Boolean, default=False)  # Whether this has been processed
    
    __table_args__ = (
        Index('idx_sync_record_table_device', 'table_name', 'device_id'),
    )
    
    def __repr__(self):
        return f"<SyncRecord {self.id} from {self.device_id}>"

# For local SQLite database on devices and in-memory on server
DB_URL = os.environ.get("database_url", "sqlite:///./sqlite.db")
logger.info(f"Using database URL: {DB_URL}")

# Add connection retries
def get_engine():
    """Creates and returns a SQLAlchemy engine"""
    retries = 3
    for attempt in range(retries):
        try:
            engine = create_engine(
                DB_URL, 
                connect_args={"check_same_thread": False},
                pool_pre_ping=True
            )
            
            # Test connection
            with engine.connect() as conn:
                pass
                
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

# Create engine and session factory
try:
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.critical(f"Fatal error initializing database engine: {e}")
    engine = None
    SessionLocal = None

def get_db() -> Generator[Session, None, None]:
    """Dependency function that provides a database session"""
    if SessionLocal is None:
        raise RuntimeError("Database connection not established")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_database():
    """Create all database tables if they don't exist"""
    if engine is None:
        logger.error("Cannot initialize database: engine is None")
        raise RuntimeError("Database engine not initialized")
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def init_db():
    """Legacy initialization function for backward compatibility"""
    warnings.warn(
        "init_db() is deprecated. Please use initialize_database() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    return initialize_database()

# Sync functions
def get_pending_sync_records(db: Session, device_id: str) -> Dict[str, List[Dict]]:
    """Get all records pending sync from this device"""
    result = {
        "users": [],
        "conversations": [],
        "chat_messages": [],
        "conversation_summaries": [],
        "devices": [],
        "messages": []
    }
    
    # Get pending sync records for each table
    pending_users = db.query(User).filter(
        User.sync_status == "pending_sync",
        User.device_id == device_id
    ).all()
    result["users"] = [user.to_dict() for user in pending_users]
    
    pending_conversations = db.query(Conversation).filter(
        Conversation.sync_status == "pending_sync",
        Conversation.device_id == device_id
    ).all()
    result["conversations"] = [conv.to_dict() for conv in pending_conversations]
    
    pending_messages = db.query(ChatMessage).filter(
        ChatMessage.sync_status == "pending_sync",
        ChatMessage.device_id == device_id
    ).all()
    result["chat_messages"] = [msg.to_dict() for msg in pending_messages]
    
    pending_summaries = db.query(ConversationSummary).filter(
        ConversationSummary.sync_status == "pending_sync",
        ConversationSummary.device_id == device_id
    ).all()
    result["conversation_summaries"] = [summary.to_dict() for summary in pending_summaries]
    
    pending_devices = db.query(Device).filter(
        Device.sync_status == "pending_sync",
        Device.device_id == device_id
    ).all()
    result["devices"] = [device.to_dict() for device in pending_devices]
    
    pending_legacy_messages = db.query(Message).filter(
        Message.sync_status == "pending_sync",
        Message.device_id == device_id
    ).all()
    result["messages"] = [msg.to_dict() for msg in pending_legacy_messages]
    
    return result

def mark_records_as_synced(db: Session, records: Dict[str, List[Dict]]):
    """Mark records as synced after successful sync"""
    # Update users
    for user_data in records.get("users", []):
        user = db.query(User).filter(User.id == uuid.UUID(user_data["id"])).first()
        if user:
            user.sync_status = "synced"
    
    # Update conversations
    for conv_data in records.get("conversations", []):
        conv = db.query(Conversation).filter(Conversation.id == uuid.UUID(conv_data["id"])).first()
        if conv:
            conv.sync_status = "synced"
    
    # Update chat messages
    for msg_data in records.get("chat_messages", []):
        msg = db.query(ChatMessage).filter(ChatMessage.id == uuid.UUID(msg_data["id"])).first()
        if msg:
            msg.sync_status = "synced"
    
    # Update conversation summaries
    for summary_data in records.get("conversation_summaries", []):
        summary = db.query(ConversationSummary).filter(ConversationSummary.id == uuid.UUID(summary_data["id"])).first()
        if summary:
            summary.sync_status = "synced"
    
    # Update devices
    for device_data in records.get("devices", []):
        device = db.query(Device).filter(Device.id == uuid.UUID(device_data["id"])).first()
        if device:
            device.sync_status = "synced"
    
    # Update legacy messages
    for msg_data in records.get("messages", []):
        msg = db.query(Message).filter(Message.id == msg_data["id"]).first()
        if msg:
            msg.sync_status = "synced"
    
    db.commit()

def process_incoming_sync_data(db: Session, sync_data: Dict[str, List[Dict]], source_device_id: str):
    """Process incoming sync data from devices"""
    # Store each record for processing
    for table_name, records in sync_data.items():
        for record in records:
            # Store in sync_records table
            sync_record = SyncRecord(
                id=record.get("id"),
                table_name=table_name,
                record_data=json.dumps(record),
                device_id=source_device_id,
                timestamp=datetime.utcnow(),
                processed=False
            )
            db.add(sync_record)
    
    db.commit()
    return {"status": "received", "record_count": sum(len(records) for records in sync_data.values())}

def get_sync_data_for_device(db: Session, device_id: str, last_sync_time: str) -> Dict[str, List[Dict]]:
    """Get sync data to send to a device"""
    # Parse the last sync time
    try:
        sync_time = datetime.fromisoformat(last_sync_time)
    except (ValueError, TypeError):
        sync_time = datetime(2000, 1, 1)  # Default to old date if invalid
    
    # Get all sync records for other devices since last sync
    sync_records = db.query(SyncRecord).filter(
        SyncRecord.device_id != device_id,
        SyncRecord.timestamp > sync_time,
        SyncRecord.processed == False
    ).all()
    
    # Organize by table name
    result = {
        "users": [],
        "conversations": [],
        "chat_messages": [],
        "conversation_summaries": [],
        "devices": [],
        "messages": []
    }
    
    for record in sync_records:
        table = record.table_name
        if table in result:
            result[table].append(json.loads(record.record_data))
            # Mark as processed
            record.processed = True
    
    db.commit()
    return result

# Automatic sync status update on record changes
@event.listens_for(Session, 'before_flush')
def set_sync_status_on_change(session, context, instances):
    """Set sync_status to 'pending_sync' when records change"""
    for obj in session.new:
        if hasattr(obj, 'sync_status'):
            obj.sync_status = 'pending_sync'
            obj.last_modified = datetime.utcnow()
    
    for obj in session.dirty:
        if hasattr(obj, 'sync_status') and session.is_modified(obj):
            obj.sync_status = 'pending_sync'
            obj.last_modified = datetime.utcnow()

def check_database_health() -> bool:
    """Checks if the database is accessible and operational"""
    if engine is None:
        return False
    
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

# Make sure all necessary items are available for import
__all__ = [
    'Base', 'engine', 'SessionLocal', 
    'User', 'Message', 'Device', 'Conversation', 'ChatMessage', 'ConversationSummary', 'SyncRecord',
    'get_db', 'initialize_database', 'init_db', 'check_database_health',
    'get_pending_sync_records', 'mark_records_as_synced', 'process_incoming_sync_data', 'get_sync_data_for_device'
]