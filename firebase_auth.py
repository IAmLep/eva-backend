"""
Firebase Authentication module for EVA backend.

Provides Firebase ID token verification for Google login.
Users authenticate via Firebase Auth on the frontend (Google sign-in),
then send the Firebase ID token to the backend for verification.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from config import settings
from database import get_db_manager
from models import User, UserInDB, UserRole
from exceptions import AuthenticationError, DatabaseError
from auth import get_password_hash

logger = logging.getLogger(__name__)

# Firebase Admin SDK is already initialized in database.py
try:
    from firebase_admin import auth as firebase_auth_admin
    FIREBASE_AUTH_AVAILABLE = True
except ImportError:
    FIREBASE_AUTH_AVAILABLE = False
    firebase_auth_admin = None
    logger.warning("Firebase Admin Auth not available. Firebase authentication disabled.")


async def verify_firebase_token(id_token: str) -> Dict[str, Any]:
    """
    Verify a Firebase ID token and return the decoded claims.
    
    Args:
        id_token: The Firebase ID token from the client.
        
    Returns:
        Dict with decoded token claims (uid, email, name, picture, etc.)
        
    Raises:
        AuthenticationError: If the token is invalid or expired.
    """
    if not FIREBASE_AUTH_AVAILABLE:
        raise AuthenticationError(detail="Firebase authentication is not configured")
    
    try:
        decoded_token = firebase_auth_admin.verify_id_token(id_token)
        return decoded_token
    except firebase_auth_admin.ExpiredIdTokenError:
        raise AuthenticationError(detail="Firebase token has expired")
    except firebase_auth_admin.RevokedIdTokenError:
        raise AuthenticationError(detail="Firebase token has been revoked")
    except firebase_auth_admin.InvalidIdTokenError:
        raise AuthenticationError(detail="Invalid Firebase token")
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
        raise AuthenticationError(detail="Could not verify Firebase token")


async def get_or_create_firebase_user(decoded_token: Dict[str, Any]) -> UserInDB:
    """
    Get an existing user by Firebase UID/email, or create a new one.
    
    This is used after Firebase token verification to ensure the user
    exists in our database.
    
    Args:
        decoded_token: Decoded Firebase ID token claims.
        
    Returns:
        UserInDB object for the authenticated user.
    """
    db = get_db_manager()
    
    email = decoded_token.get("email")
    firebase_uid = decoded_token.get("uid")
    display_name = decoded_token.get("name", "")
    picture = decoded_token.get("picture", "")
    
    if not email:
        raise AuthenticationError(detail="Firebase token does not contain an email")
    
    # Try to find existing user by email
    try:
        existing_user = await db.get_user_by_email(email)
        if existing_user:
            # Update metadata with latest Firebase info if needed
            updates = {
                "metadata": {
                    **(existing_user.metadata or {}),
                    "firebase_uid": firebase_uid,
                    "picture": picture,
                },
            }
            if display_name and not existing_user.full_name:
                updates["full_name"] = display_name
            await db.update_user(existing_user.id, updates)
            return existing_user
    except DatabaseError as e:
        logger.error(f"Database error looking up user by email {email}: {e}")
        raise
    
    # Create new user from Firebase auth info
    # Use email prefix as username, ensuring uniqueness
    username = email.split("@")[0]
    base_username = username
    counter = 1
    
    while True:
        existing = await db.get_user_by_username(username)
        if not existing:
            break
        username = f"{base_username}{counter}"
        counter += 1
        if counter > 100:
            raise DatabaseError("Could not generate unique username")
    
    # Create a placeholder hashed password (user authenticates via Firebase, not password)
    placeholder_password = get_password_hash(str(uuid.uuid4()))
    
    new_user = UserInDB(
        id=str(uuid.uuid4()),
        username=username,
        email=email,
        full_name=display_name or username,
        hashed_password=placeholder_password,
        disabled=False,
        role=UserRole.USER,
        preferences={},
        metadata={
            "firebase_uid": firebase_uid,
            "picture": picture,
            "auth_provider": "google",
        },
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    
    try:
        user_id = await db.create_user(new_user)
        if not user_id:
            raise DatabaseError("Failed to create user from Firebase auth")
        
        # Retrieve the created user
        created_user = await db.get_user(user_id)
        if not created_user:
            raise DatabaseError("User created but could not be retrieved")
        
        logger.info(f"Created new user from Firebase auth: {email} (ID: {user_id})")
        return created_user
        
    except DatabaseError:
        raise
    except Exception as e:
        logger.error(f"Error creating user from Firebase auth: {e}")
        raise DatabaseError(f"Failed to create user: {e}")
