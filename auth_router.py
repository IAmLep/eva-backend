"""
Authentication Router module for EVA backend.

This module provides API endpoints for authentication-related operations
including user registration, login, token management, and verification.

"""
"""
Version 3 working
"""

import logging
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator

from auth import Token, create_access_token, get_current_active_user, get_password_hash
from config import get_settings
from database import get_db_manager
from exceptions import AuthenticationError, DatabaseError
from models import User, UserInDB

# Setup router
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)


class UserCreate(BaseModel):
    """
    User creation request model.
    
    Attributes:
        username: Username (must be unique)
        email: Email address
        password: Password (will be hashed)
        full_name: Optional full name
    """
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        # Could add more validation rules here (uppercase, lowercase, numbers, etc.)
        return v
    
    @validator('username')
    def username_valid(cls, v):
        """Validate username format."""
        if not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        return v


class UserResponse(BaseModel):
    """
    User response model (excludes sensitive information).
    
    Attributes:
        id: User ID
        username: Username
        email: Email address
        full_name: Optional full name
        created_at: Creation timestamp
        is_active: Whether user is active
    """
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    created_at: datetime
    is_active: bool = True


class LoginResponse(BaseModel):
    """
    Login response model including token and user information.
    
    Attributes:
        access_token: JWT access token
        token_type: Type of token (always 'bearer')
        expires_at: Timestamp when token expires
        user: User information
    """
    access_token: str
    token_type: str
    expires_at: datetime
    user: UserResponse


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate) -> Dict:
    """
    Register a new user.
    
    Args:
        user_data: User creation data
        
    Returns:
        Dict: Created user information
        
    Raises:
        HTTPException: If registration fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Check if username already exists
        existing_user = await db.get_user_by_username(user_data.username)
        if existing_user:
            logger.warning(f"Registration attempt with existing username: {user_data.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Create new user with hashed password
        hashed_password = get_password_hash(user_data.password)
        new_user = UserInDB(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            created_at=datetime.utcnow(),
            is_active=True,
            disabled=False
        )
        
        # Add user to database
        user_id = await db.create_user(new_user)
        
        # Retrieve the created user (with ID)
        created_user = await db.get_user_by_id(user_id)
        if not created_user:
            logger.error(f"User created but not found: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User created but retrieval failed"
            )
        
        logger.info(f"Successfully registered new user: {user_data.username}")
        
        # Return user without sensitive information
        return {
            "id": user_id,
            "username": created_user.username,
            "email": created_user.email,
            "full_name": created_user.full_name,
            "created_at": created_user.created_at,
            "is_active": not created_user.disabled
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=LoginResponse)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Dict:
    """
    Login user and get access token.
    
    Args:
        form_data: OAuth2 form with username and password
        
    Returns:
        Dict: Login response with token and user information
        
    Raises:
        HTTPException: If login fails
    """
    try:
        # Use the token endpoint from auth.py
        from auth import login_for_access_token
        
        # Get token
        token_response = await login_for_access_token(form_data)
        
        # Get user information
        db = get_db_manager()
        user = await db.get_user_by_username(form_data.username)
        
        if not user:
            logger.error(f"User not found after successful token generation: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User retrieval failed"
            )
        
        logger.info(f"User logged in successfully: {form_data.username}")
        
        # Return token and user information
        return {
            **token_response,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "created_at": user.created_at,
                "is_active": not user.disabled
            }
        }
    
    except AuthenticationError as e:
        logger.warning(f"Login failed due to authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.get("/me", response_model=UserResponse)
async def get_user_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict:
    """
    Get current authenticated user information.
    
    Args:
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: User information
    """
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "created_at": current_user.created_at,
        "is_active": not current_user.disabled
    }


@router.put("/me", response_model=UserResponse)
async def update_user_me(
    user_update: Dict,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict:
    """
    Update current authenticated user information.
    
    Args:
        user_update: Fields to update
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: Updated user information
        
    Raises:
        HTTPException: If update fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Handle password update separately
        if "password" in user_update:
            user_update["hashed_password"] = get_password_hash(user_update.pop("password"))
        
        # Don't allow updating username or email directly
        if "username" in user_update:
            logger.warning(f"Username update attempt for user {current_user.id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username cannot be changed"
            )
        
        # Update user
        success = await db.update_user(current_user.id, user_update)
        
        if not success:
            logger.error(f"Failed to update user {current_user.id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
        
        # Get updated user
        updated_user = await db.get_user_by_id(current_user.id)
        
        logger.info(f"User {current_user.username} updated successfully")
        
        # Return updated user information
        return {
            "id": updated_user.id,
            "username": updated_user.username,
            "email": updated_user.email,
            "full_name": updated_user.full_name,
            "created_at": updated_user.created_at,
            "is_active": not updated_user.disabled
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {str(e)}"
        )


@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict:
    """
    Refresh access token.
    
    Args:
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: New token information
    """
    settings = get_settings()
    
    # Calculate expiration time
    expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.utcnow() + expires_delta
    
    # Create new token
    access_token = create_access_token(
        data={"sub": current_user.username},
        expires_delta=expires_delta
    )
    
    logger.info(f"Token refreshed for user: {current_user.username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": expires_at
    }


@router.post("/logout")
async def logout(
    response: Response,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict:
    """
    Logout user.
    
    Note: JWT tokens cannot be invalidated server-side.
    This endpoint is mainly for client-side cleanup purposes.
    
    Args:
        response: FastAPI response
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: Logout confirmation
    """
    # Log the logout event
    logger.info(f"User logged out: {current_user.username}")
    
    # Set an expired cookie to help client clear the token
    response.delete_cookie(key="access_token")
    
    return {"message": "Successfully logged out"}


@router.post("/change-password")
async def change_password(
    password_data: Dict,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict:
    """
    Change user password.
    
    Args:
        password_data: Password change data including old_password and new_password
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: Password change confirmation
        
    Raises:
        HTTPException: If password change fails
    """
    try:
        # Validate request
        if "old_password" not in password_data or "new_password" not in password_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both old_password and new_password are required"
            )
        
        # Validate new password
        if len(password_data["new_password"]) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters long"
            )
        
        # Verify old password
        from auth import verify_password
        db = get_db_manager()
        user = await db.get_user_by_username(current_user.username)
        
        if not verify_password(password_data["old_password"], user.hashed_password):
            logger.warning(f"Password change attempt with incorrect old password: {current_user.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect old password"
            )
        
        # Update password
        new_hashed_password = get_password_hash(password_data["new_password"])
        success = await db.update_user(
            current_user.id, 
            {"hashed_password": new_hashed_password}
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        
        logger.info(f"Password changed successfully for user: {current_user.username}")
        
        return {"message": "Password changed successfully"}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )


@router.get("/verify-token")
async def verify_token(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict:
    """
    Verify if access token is valid.
    
    Args:
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: Token verification result
    """
    return {
        "valid": True,
        "username": current_user.username
    }