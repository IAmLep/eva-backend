# Replace this section in auth.py:

# Redis connection for token blacklist
try:
    from redis_manager import get_redis_client
    redis_client = get_redis_client()
except Exception as e:
    print(f"Warning: Redis connection failed: {e}")
    redis_client = None

# With this Firestore implementation:

# Firestore for token blacklist
try:
    from firestore_manager import store_document, get_document
    blacklist_collection = "token_blacklist"
except Exception as e:
    print(f"Warning: Firestore import failed: {e}")
    # Fall back to in-memory only
    
# Then replace these functions:

def is_token_blacklisted(token_jti: str) -> bool:
    """
    Check if a token is blacklisted using Firestore.
    
    Args:
        token_jti: JWT ID to check
        
    Returns:
        True if blacklisted, False otherwise
    """
    try:
        # Use asyncio.run in synchronous context
        import asyncio
        result = asyncio.run(get_document(blacklist_collection, token_jti))
        return bool(result)
    except Exception as e:
        print(f"Error checking token blacklist: {e}")
        # Fall back to in-memory set
        return token_jti in revoked_tokens

def blacklist_token(token_jti: str, expires_in: int = None):
    """
    Add a token to the blacklist using Firestore.
    
    Args:
        token_jti: JWT ID to blacklist
        expires_in: Seconds until token expires naturally
    """
    try:
        # Use asyncio.run in synchronous context
        import asyncio
        data = {"revoked_at": datetime.utcnow().isoformat()}
        
        if expires_in:
            data["expires_at"] = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
            
        asyncio.run(store_document(blacklist_collection, token_jti, data))
    except Exception as e:
        print(f"Error adding token to blacklist: {e}")
        # Fall back to in-memory set
        revoked_tokens.add(token_jti)