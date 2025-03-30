"""
Firestore client manager for Eva backend.
Provides functions for interacting with Firestore collections.
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from functools import wraps

from google.cloud import firestore
from config import FIRESTORE_COLLECTION_PREFIX

logger = logging.getLogger(__name__)

# Singleton firestore client
_firestore_client = None

def get_firestore_client() -> firestore.Client:
    """Get or create the Firestore client."""
    global _firestore_client
    if _firestore_client is None:
        try:
            _firestore_client = firestore.Client()
            logger.info("Firestore client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            raise
    return _firestore_client

def _get_collection_name(collection: str) -> str:
    """Add prefix to collection name if configured."""
    if FIRESTORE_COLLECTION_PREFIX:
        return f"{FIRESTORE_COLLECTION_PREFIX}{collection}"
    return collection

def _handle_firestore_errors(func):
    """Decorator to handle Firestore errors."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Firestore operation failed: {func.__name__} - {str(e)}")
            return None if "get" in func.__name__ else False
    return wrapper

@_handle_firestore_errors
async def store_document(collection: str, document_id: str, data: Dict[str, Any]) -> bool:
    """Store a document in Firestore."""
    client = get_firestore_client()
    collection_name = _get_collection_name(collection)
    
    # Add metadata
    data["updated_at"] = datetime.utcnow().isoformat()
    if "created_at" not in data:
        data["created_at"] = datetime.utcnow().isoformat()
    
    # Run in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: client.collection(collection_name).document(document_id).set(data)
    )
    
    logger.debug(f"Stored document in {collection_name}/{document_id}")
    return True

@_handle_firestore_errors
async def get_document(collection: str, document_id: str) -> Optional[Dict[str, Any]]:
    """Get a document from Firestore."""
    client = get_firestore_client()
    collection_name = _get_collection_name(collection)
    
    # Run in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    doc_snapshot = await loop.run_in_executor(
        None, 
        lambda: client.collection(collection_name).document(document_id).get()
    )
    
    if doc_snapshot.exists:
        return doc_snapshot.to_dict()
    return None

@_handle_firestore_errors
async def update_document(collection: str, document_id: str, data: Dict[str, Any]) -> bool:
    """Update specific fields in a document."""
    client = get_firestore_client()
    collection_name = _get_collection_name(collection)
    
    # Add updated timestamp
    data["updated_at"] = datetime.utcnow().isoformat()
    
    # Run in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: client.collection(collection_name).document(document_id).update(data)
    )
    
    logger.debug(f"Updated document {collection_name}/{document_id}")
    return True

@_handle_firestore_errors
async def delete_document(collection: str, document_id: str) -> bool:
    """Delete a document from Firestore."""
    client = get_firestore_client()
    collection_name = _get_collection_name(collection)
    
    # Run in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: client.collection(collection_name).document(document_id).delete()
    )
    
    logger.debug(f"Deleted document {collection_name}/{document_id}")
    return True

@_handle_firestore_errors
async def add_to_collection(collection: str, data: Dict[str, Any]) -> Optional[str]:
    """Add a document to a collection with auto-generated ID."""
    client = get_firestore_client()
    collection_name = _get_collection_name(collection)
    
    # Add metadata
    data["created_at"] = datetime.utcnow().isoformat()
    data["updated_at"] = datetime.utcnow().isoformat()
    
    # Run in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    doc_ref = await loop.run_in_executor(
        None,
        lambda: client.collection(collection_name).add(data)[1]
    )
    
    logger.debug(f"Added document to {collection_name} with ID {doc_ref.id}")
    return doc_ref.id

@_handle_firestore_errors
async def query_collection(
    collection: str, 
    filters: List[Tuple[str, str, Any]], 
    limit: int = 100,
    order_by: Optional[str] = None,
    direction: str = "DESCENDING"
) -> List[Dict[str, Any]]:
    """
    Query documents in a collection with multiple filters.
    
    Args:
        collection: Name of the collection
        filters: List of (field, operator, value) tuples
        limit: Maximum number of results
        order_by: Field to order by
        direction: "ASCENDING" or "DESCENDING"
        
    Returns:
        List of documents matching the query
    """
    client = get_firestore_client()
    collection_name = _get_collection_name(collection)
    
    # Create query
    query = client.collection(collection_name)
    
    # Add filters
    for field, operator, value in filters:
        query = query.where(field, operator, value)
    
    # Add ordering
    if order_by:
        direction_obj = firestore.Query.DESCENDING if direction == "DESCENDING" else firestore.Query.ASCENDING
        query = query.order_by(order_by, direction=direction_obj)
    
    # Add limit
    query = query.limit(limit)
    
    # Run query in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: query.stream())
    
    return [doc.to_dict() for doc in results]

@_handle_firestore_errors
async def document_exists(collection: str, document_id: str) -> bool:
    """Check if a document exists."""
    client = get_firestore_client()
    collection_name = _get_collection_name(collection)
    
    # Run in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    doc_snapshot = await loop.run_in_executor(
        None, 
        lambda: client.collection(collection_name).document(document_id).get()
    )
    
    return doc_snapshot.exists

# Specialized functions for authentication

async def store_with_expiry(collection: str, doc_id: str, data: Dict[str, Any], 
                           expires_in_seconds: int) -> bool:
    """Store a document with expiration time."""
    expiry = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
    data["expires_at"] = expiry.isoformat()
    return await store_document(collection, doc_id, data)

async def get_if_not_expired(collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """Get a document if it hasn't expired."""
    data = await get_document(collection, doc_id)
    if not data or "expires_at" not in data:
        return None
    
    expires_at = datetime.fromisoformat(data["expires_at"])
    if datetime.utcnow() > expires_at:
        # Expired - delete it
        await delete_document(collection, doc_id)
        return None
    
    return data

async def get_rate_limit(key: str) -> Tuple[int, datetime]:
    """Get current rate limit count and expiry for a key."""
    data = await get_document("rate_limits", key)
    if not data:
        return 0, datetime.utcnow()
    
    count = data.get("count", 0)
    expires_at = datetime.fromisoformat(data.get("expires_at"))
    return count, expires_at

async def update_rate_limit(key: str, count: int, expires_at: datetime) -> bool:
    """Update rate limit data."""
    return await store_document("rate_limits", key, {
        "count": count,
        "expires_at": expires_at.isoformat()
    })