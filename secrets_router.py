"""
Secrets Router module for EVA backend.

This module provides API endpoints for secure management of user secrets
and sensitive information with encryption and access controls.


Version 3 working
"""

import base64
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from pydantic import BaseModel, Field, validator

from auth import get_current_active_user
from config import get_settings
from database import get_db_manager
from exceptions import AuthorizationError, DatabaseError, NotFoundException
from logging_config import get_logger
from models import User
from rate_limiter import rate_limit, RateLimitType, RateLimitWindow

# Setup router
router = APIRouter()

# Logger configuration
logger = get_logger(__name__)


class SecretCategory(BaseModel):
    """
    Secret category model.
    
    Represents a category for organizing secrets.
    
    Attributes:
        id: Unique identifier
        name: Category name
        description: Optional category description
        icon: Optional icon identifier
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None


class SecretCreateRequest(BaseModel):
    """
    Secret creation request model.
    
    Attributes:
        name: Secret name
        value: Secret value to encrypt
        description: Optional description
        category_id: Optional category identifier
        metadata: Optional additional metadata
        tags: Optional tags for filtering
    """
    name: str
    value: str
    description: Optional[str] = None
    category_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    @field_validator('name')
    def name_not_empty(cls, v):
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v
    
    @field_validator('value')
    def value_not_empty(cls, v):
        """Validate that value is not empty."""
        if not v:
            raise ValueError('Value cannot be empty')
        return v


class SecretUpdateRequest(BaseModel):
    """
    Secret update request model.
    
    Attributes:
        name: Optional new name
        value: Optional new value to encrypt
        description: Optional new description
        category_id: Optional new category identifier
        metadata: Optional new metadata
        tags: Optional new tags
    """
    name: Optional[str] = None
    value: Optional[str] = None
    description: Optional[str] = None
    category_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    @field_validator('name')
    def name_not_empty(cls, v):
        """Validate that name is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError('Name cannot be empty')
        return v
    
    @field_validator('value')
    def value_not_empty(cls, v):
        """Validate that value is not empty if provided."""
        if v is not None and not v:
            raise ValueError('Value cannot be empty')
        return v


class SecretResponse(BaseModel):
    """
    Secret response model.
    
    Attributes:
        id: Secret identifier
        name: Secret name
        description: Optional description
        category_id: Optional category identifier
        metadata: Additional metadata
        tags: Tags for filtering
        created_at: Creation timestamp
        updated_at: Last update timestamp
        has_value: Whether the secret has a value
    """
    id: str
    name: str
    description: Optional[str] = None
    category_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    has_value: bool


class SecretWithValueResponse(SecretResponse):
    """
    Secret response model with decrypted value.
    
    Extends SecretResponse with the decrypted value.
    
    Attributes:
        value: Decrypted secret value
    """
    value: str


class CategoryCreateRequest(BaseModel):
    """
    Category creation request model.
    
    Attributes:
        name: Category name
        description: Optional description
        icon: Optional icon identifier
    """
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    
    @field_validator('name')
    def name_not_empty(cls, v):
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v


class SecretsEncryptor:
    """
    Utility class for encrypting and decrypting secrets.
    
    Uses Fernet symmetric encryption with a key derived from the app secret.
    """
    
    def __init__(self):
        """Initialize the secrets encryptor with key derived from settings."""
        self.settings = get_settings()
        self._fernet = self._create_fernet()
    
    def _create_fernet(self) -> Fernet:
        """
        Create Fernet instance with key derived from app secret.
        
        Returns:
            Fernet: Initialized Fernet instance
        """
        # Get master key from settings
        master_key = self.settings.SECRET_KEY
        
        # Use a static salt for key derivation
        # In production, this should be a securely stored value
        salt = b'eva_secrets_salt_2025'
        
        # Derive a key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        # Generate the key
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        
        # Create and return Fernet instance
        return Fernet(key)
    
    def encrypt(self, value: str) -> str:
        """
        Encrypt a value.
        
        Args:
            value: Value to encrypt
            
        Returns:
            str: Base64-encoded encrypted value
        """
        # Encrypt the value
        encrypted = self._fernet.encrypt(value.encode())
        
        # Return as base64 string
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_value: str) -> str:
        """
        Decrypt a value.
        
        Args:
            encrypted_value: Base64-encoded encrypted value
            
        Returns:
            str: Decrypted value
            
        Raises:
            ValueError: If decryption fails
        """
        try:
            # Decode from base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            
            # Decrypt the value
            decrypted = self._fernet.decrypt(encrypted_bytes)
            
            # Return as string
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise ValueError(f"Failed to decrypt value: {str(e)}")


# Singleton instance of the encryptor
_secrets_encryptor: Optional[SecretsEncryptor] = None


def get_secrets_encryptor() -> SecretsEncryptor:
    """
    Get secrets encryptor singleton.
    
    Returns:
        SecretsEncryptor: Secrets encryptor instance
    """
    global _secrets_encryptor
    if _secrets_encryptor is None:
        _secrets_encryptor = SecretsEncryptor()
    return _secrets_encryptor


@router.post("", response_model=SecretResponse, status_code=status.HTTP_201_CREATED)
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.MINUTE, increment=1)
async def create_secret(
    secret: SecretCreateRequest,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Create a new secret.
    
    Args:
        secret: Secret creation request
        current_user: Current authenticated user
        
    Returns:
        Dict: Created secret information
        
    Raises:
        HTTPException: If secret creation fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Generate UUID for the secret
        secret_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Get encryptor
        encryptor = get_secrets_encryptor()
        
        # Encrypt the secret value
        encrypted_value = encryptor.encrypt(secret.value)
        
        # Create secret record
        secret_data = {
            "id": secret_id,
            "user_id": current_user.id,
            "name": secret.name,
            "encrypted_value": encrypted_value,
            "description": secret.description,
            "category_id": secret.category_id,
            "metadata": secret.metadata,
            "tags": secret.tags,
            "created_at": now,
            "updated_at": now
        }
        
        # Check category if provided
        if secret.category_id:
            category = await db.get_category(current_user.id, secret.category_id)
            if not category:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Category with ID {secret.category_id} not found"
                )
        
        # Store secret
        await db.create_secret(secret_data)
        
        logger.info(f"Secret created: {secret_id} for user {current_user.id}")
        
        # Return response without the encrypted value
        return {
            "id": secret_id,
            "name": secret.name,
            "description": secret.description,
            "category_id": secret.category_id,
            "metadata": secret.metadata,
            "tags": secret.tags,
            "created_at": now,
            "updated_at": now,
            "has_value": True
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating secret: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create secret: {str(e)}"
        )


@router.get("", response_model=List[SecretResponse])
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.MINUTE, increment=1)
async def get_secrets(
    category_id: Optional[str] = None,
    tag: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> List[Dict]:
    """
    Get all secrets for the current user.
    
    Args:
        category_id: Optional category ID filter
        tag: Optional tag filter
        current_user: Current authenticated user
        
    Returns:
        List[Dict]: List of secrets
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get secrets
        secrets = await db.get_user_secrets(
            user_id=current_user.id,
            category_id=category_id,
            tag=tag
        )
        
        # Convert to response format
        response_secrets = []
        for secret in secrets:
            response_secrets.append({
                "id": secret["id"],
                "name": secret["name"],
                "description": secret["description"],
                "category_id": secret["category_id"],
                "metadata": secret["metadata"],
                "tags": secret["tags"],
                "created_at": secret["created_at"],
                "updated_at": secret["updated_at"],
                "has_value": "encrypted_value" in secret and bool(secret["encrypted_value"])
            })
        
        logger.info(f"Retrieved {len(response_secrets)} secrets for user {current_user.id}")
        return response_secrets
    
    except Exception as e:
        logger.error(f"Error retrieving secrets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve secrets: {str(e)}"
        )


@router.get("/{secret_id}", response_model=SecretWithValueResponse)
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.MINUTE, increment=1)
async def get_secret(
    secret_id: str,
    include_value: bool = True,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Get a specific secret by ID.
    
    Args:
        secret_id: Secret ID to retrieve
        include_value: Whether to include the decrypted value
        current_user: Current authenticated user
        
    Returns:
        Dict: Secret information
        
    Raises:
        HTTPException: If retrieval fails or secret not found
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get secret
        secret = await db.get_secret(secret_id)
        
        if not secret:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Secret with ID {secret_id} not found"
            )
        
        # Check ownership
        if secret["user_id"] != current_user.id:
            logger.warning(f"User {current_user.id} attempted to access secret {secret_id} owned by {secret['user_id']}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this secret"
            )
        
        # Prepare response
        response = {
            "id": secret["id"],
            "name": secret["name"],
            "description": secret["description"],
            "category_id": secret["category_id"],
            "metadata": secret["metadata"],
            "tags": secret["tags"],
            "created_at": secret["created_at"],
            "updated_at": secret["updated_at"],
            "has_value": "encrypted_value" in secret and bool(secret["encrypted_value"])
        }
        
        # Decrypt and include value if requested
        if include_value and "encrypted_value" in secret and secret["encrypted_value"]:
            try:
                encryptor = get_secrets_encryptor()
                response["value"] = encryptor.decrypt(secret["encrypted_value"])
            except ValueError as e:
                logger.error(f"Error decrypting secret {secret_id}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to decrypt secret value: {str(e)}"
                )
        elif include_value:
            # Include empty value if requested but not available
            response["value"] = ""
        
        logger.info(f"Retrieved secret {secret_id} for user {current_user.id}")
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving secret {secret_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve secret: {str(e)}"
        )


@router.put("/{secret_id}", response_model=SecretResponse)
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.MINUTE, increment=1)
async def update_secret(
    secret_id: str,
    secret_update: SecretUpdateRequest,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Update a secret.
    
    Args:
        secret_id: Secret ID to update
        secret_update: Secret update request
        current_user: Current authenticated user
        
    Returns:
        Dict: Updated secret information
        
    Raises:
        HTTPException: If update fails or secret not found
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get existing secret
        existing_secret = await db.get_secret(secret_id)
        
        if not existing_secret:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Secret with ID {secret_id} not found"
            )
        
        # Check ownership
        if existing_secret["user_id"] != current_user.id:
            logger.warning(f"User {current_user.id} attempted to update secret {secret_id} owned by {existing_secret['user_id']}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this secret"
            )
        
        # Check category if provided
        if secret_update.category_id:
            category = await db.get_category(current_user.id, secret_update.category_id)
            if not category:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Category with ID {secret_update.category_id} not found"
                )
        
        # Prepare update data
        update_data = {}
        for field, value in secret_update.dict(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value
        
        # Encrypt value if provided
        if "value" in update_data:
            encryptor = get_secrets_encryptor()
            update_data["encrypted_value"] = encryptor.encrypt(update_data.pop("value"))
        
        # Set updated timestamp
        update_data["updated_at"] = datetime.utcnow()
        
        # Update secret
        success = await db.update_secret(secret_id, update_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update secret"
            )
        
        # Get updated secret
        updated_secret = await db.get_secret(secret_id)
        
        logger.info(f"Updated secret {secret_id} for user {current_user.id}")
        
        # Return response
        return {
            "id": updated_secret["id"],
            "name": updated_secret["name"],
            "description": updated_secret["description"],
            "category_id": updated_secret["category_id"],
            "metadata": updated_secret["metadata"],
            "tags": updated_secret["tags"],
            "created_at": updated_secret["created_at"],
            "updated_at": updated_secret["updated_at"],
            "has_value": "encrypted_value" in updated_secret and bool(updated_secret["encrypted_value"])
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error updating secret {secret_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update secret: {str(e)}"
        )


@router.delete("/{secret_id}")
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.MINUTE, increment=1)
async def delete_secret(
    secret_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Delete a secret.
    
    Args:
        secret_id: Secret ID to delete
        current_user: Current authenticated user
        
    Returns:
        Dict: Deletion confirmation
        
    Raises:
        HTTPException: If deletion fails or secret not found
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get existing secret
        existing_secret = await db.get_secret(secret_id)
        
        if not existing_secret:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Secret with ID {secret_id} not found"
            )
        
        # Check ownership
        if existing_secret["user_id"] != current_user.id:
            logger.warning(f"User {current_user.id} attempted to delete secret {secret_id} owned by {existing_secret['user_id']}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this secret"
            )
        
        # Delete secret
        success = await db.delete_secret(secret_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete secret"
            )
        
        logger.info(f"Deleted secret {secret_id} for user {current_user.id}")
        
        return {"success": True, "message": "Secret deleted successfully"}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting secret {secret_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete secret: {str(e)}"
        )


@router.post("/categories", response_model=SecretCategory, status_code=status.HTTP_201_CREATED)
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.MINUTE, increment=1)
async def create_category(
    category: CategoryCreateRequest,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Create a new category.
    
    Args:
        category: Category creation request
        current_user: Current authenticated user
        
    Returns:
        Dict: Created category information
        
    Raises:
        HTTPException: If category creation fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Generate UUID for the category
        category_id = str(uuid.uuid4())
        
        # Create category record
        category_data = {
            "id": category_id,
            "user_id": current_user.id,
            "name": category.name,
            "description": category.description,
            "icon": category.icon,
            "created_at": datetime.utcnow()
        }
        
        # Store category
        await db.create_category(category_data)
        
        logger.info(f"Category created: {category_id} for user {current_user.id}")
        
        # Return response
        return {
            "id": category_id,
            "name": category.name,
            "description": category.description,
            "icon": category.icon
        }
    
    except Exception as e:
        logger.error(f"Error creating category: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create category: {str(e)}"
        )


@router.get("/categories", response_model=List[SecretCategory])
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.MINUTE, increment=1)
async def get_categories(
    current_user: User = Depends(get_current_active_user)
) -> List[Dict]:
    """
    Get all categories for the current user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List[Dict]: List of categories
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get categories
        categories = await db.get_user_categories(current_user.id)
        
        logger.info(f"Retrieved {len(categories)} categories for user {current_user.id}")
        
        # Convert to response format
        return categories
    
    except Exception as e:
        logger.error(f"Error retrieving categories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve categories: {str(e)}"
        )


@router.delete("/categories/{category_id}")
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.MINUTE, increment=1)
async def delete_category(
    category_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Delete a category.
    
    Args:
        category_id: Category ID to delete
        current_user: Current authenticated user
        
    Returns:
        Dict: Deletion confirmation
        
    Raises:
        HTTPException: If deletion fails or category not found
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get existing category
        existing_category = await db.get_category(current_user.id, category_id)
        
        if not existing_category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category with ID {category_id} not found"
            )
        
        # Check ownership
        if existing_category["user_id"] != current_user.id:
            logger.warning(f"User {current_user.id} attempted to delete category {category_id} owned by {existing_category['user_id']}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this category"
            )
        
        # Check if category has secrets
        secrets = await db.get_user_secrets(
            user_id=current_user.id,
            category_id=category_id
        )
        
        if secrets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete category with {len(secrets)} secrets. Remove or reassign secrets first."
            )
        
        # Delete category
        success = await db.delete_category(category_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete category"
            )
        
        logger.info(f"Deleted category {category_id} for user {current_user.id}")
        
        return {"success": True, "message": "Category deleted successfully"}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting category {category_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete category: {str(e)}"
        )


@router.post("/bulk-export", response_model=Dict[str, Any])
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.DAY, increment=1)
async def export_secrets(
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Export all user secrets with their decrypted values.
    
    This is a sensitive operation that should be used with caution.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Dict: Exported secrets and categories
        
    Raises:
        HTTPException: If export fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get secrets
        secrets = await db.get_user_secrets(user_id=current_user.id)
        
        # Get categories
        categories = await db.get_user_categories(current_user.id)
        
        # Get encryptor
        encryptor = get_secrets_encryptor()
        
        # Decrypt secrets
        decrypted_secrets = []
        for secret in secrets:
            secret_copy = secret.copy()
            
            # Decrypt value if present
            if "encrypted_value" in secret_copy and secret_copy["encrypted_value"]:
                try:
                    secret_copy["value"] = encryptor.decrypt(secret_copy["encrypted_value"])
                except ValueError:
                    secret_copy["value"] = ""
                
                # Remove encrypted value
                del secret_copy["encrypted_value"]
            else:
                secret_copy["value"] = ""
            
            decrypted_secrets.append(secret_copy)
        
        logger.info(f"Exported {len(decrypted_secrets)} secrets for user {current_user.id}")
        
        # Return exported data
        return {
            "secrets": decrypted_secrets,
            "categories": categories,
            "exported_at": datetime.utcnow(),
            "user_id": current_user.id
        }
    
    except Exception as e:
        logger.error(f"Error exporting secrets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export secrets: {str(e)}"
        )


@router.post("/bulk-import", status_code=status.HTTP_201_CREATED)
@rate_limit(limit_type=RateLimitType.REQUESTS, window=RateLimitWindow.DAY, increment=1)
async def import_secrets(
    import_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Import secrets and categories.
    
    Args:
        import_data: Data to import containing secrets and categories
        current_user: Current authenticated user
        
    Returns:
        Dict: Import results
        
    Raises:
        HTTPException: If import fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get encryptor
        encryptor = get_secrets_encryptor()
        
        # Process categories first
        categories_imported = 0
        if "categories" in import_data and isinstance(import_data["categories"], list):
            for category in import_data["categories"]:
                # Skip if missing required fields
                if not isinstance(category, dict) or "name" not in category:
                    continue
                
                # Generate new ID
                category_id = str(uuid.uuid4())
                
                # Create category record
                category_data = {
                    "id": category_id,
                    "user_id": current_user.id,
                    "name": category["name"],
                    "description": category.get("description"),
                    "icon": category.get("icon"),
                    "created_at": datetime.utcnow()
                }
                
                # Store category
                await db.create_category(category_data)
                categories_imported += 1
        
        # Process secrets
        secrets_imported = 0
        if "secrets" in import_data and isinstance(import_data["secrets"], list):
            for secret in import_data["secrets"]:
                # Skip if missing required fields
                if not isinstance(secret, dict) or "name" not in secret or "value" not in secret:
                    continue
                
                # Generate new ID
                secret_id = str(uuid.uuid4())
                now = datetime.utcnow()
                
                # Encrypt the secret value
                encrypted_value = encryptor.encrypt(secret["value"])
                
                # Create secret record
                secret_data = {
                    "id": secret_id,
                    "user_id": current_user.id,
                    "name": secret["name"],
                    "encrypted_value": encrypted_value,
                    "description": secret.get("description"),
                    "category_id": None,  # Don't import category relationship directly
                    "metadata": secret.get("metadata", {}),
                    "tags": secret.get("tags", []),
                    "created_at": now,
                    "updated_at": now
                }
                
                # Store secret
                await db.create_secret(secret_data)
                secrets_imported += 1
        
        logger.info(f"Imported {secrets_imported} secrets and {categories_imported} categories for user {current_user.id}")
        
        return {
            "success": True,
            "secrets_imported": secrets_imported,
            "categories_imported": categories_imported
        }
    
    except Exception as e:
        logger.error(f"Error importing secrets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import secrets: {str(e)}"
        )