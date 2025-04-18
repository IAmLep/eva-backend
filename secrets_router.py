"""
API Router for managing user secrets and categories.

Provides endpoints for CRUD operations on secret categories and individual secrets,
ensuring user ownership and authentication.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

# --- Local Imports ---
from auth import get_current_active_user # Authentication dependency
from database import get_db_manager, DatabaseManager # Use the DB manager
from exceptions import DatabaseError, NotFoundException, AuthorizationError # Custom exceptions
from models import User # User model

# --- Router Setup ---
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)

# --- Pydantic Models for Secrets ---
# (Could reside in schemas.py, but kept here for module focus)

class SecretCategoryCreate(BaseModel):
    """Request model for creating a secret category."""
    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=200)
    icon: Optional[str] = Field(None, max_length=50) # e.g., FontAwesome name

class SecretCategoryResponse(BaseModel):
    """Response model for secret category."""
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class SecretCreate(BaseModel):
    """Request model for creating a secret."""
    category_id: str
    name: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., description="The actual secret value (will be encrypted)")
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class SecretUpdate(BaseModel):
    """Request model for updating a secret."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    value: Optional[str] = None # Allow updating the secret value
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    category_id: Optional[str] = None # Allow moving secret to different category

class SecretResponse(BaseModel):
    """Response model for a secret (value is NOT included)."""
    id: str
    user_id: str
    category_id: str
    name: str
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    # Exclude 'value' for security

# --- Encryption Placeholder ---
# TODO: Implement actual encryption/decryption for secret values
# Use a strong library like 'cryptography' with Fernet symmetric encryption.
# The key should be derived securely from user password or a master key.
# For simplicity, this is just a placeholder.
def encrypt_value(value: str, user_key: str) -> str:
    logger.warning("Using placeholder encryption (Base64). Replace with real encryption.")
    # Placeholder: Base64 encode (NOT SECURE!)
    import base64
    return base64.b64encode(value.encode()).decode()

def decrypt_value(encrypted_value: str, user_key: str) -> str:
    logger.warning("Using placeholder decryption (Base64). Replace with real encryption.")
    # Placeholder: Base64 decode
    import base64
    try:
        return base64.b64decode(encrypted_value.encode()).decode()
    except Exception as e:
        logger.error(f"Placeholder decryption failed: {e}")
        return "[DECRYPTION FAILED]"

# --- Helper Function ---
async def get_user_key_placeholder(user: User) -> str:
    """Placeholder function to get user's encryption key."""
    # In a real implementation, this would involve deriving the key securely.
    # NEVER store the raw key directly.
    logger.warning("Using placeholder user encryption key.")
    return f"key_for_{user.id}" # Highly insecure placeholder


# --- Category Endpoints ---

@router.post(
    "/categories",
    response_model=SecretCategoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new secret category"
)
async def create_secret_category(
    category_data: SecretCategoryCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: DatabaseManager = Depends(get_db_manager)
) -> SecretCategoryResponse:
    """Creates a new category for organizing secrets belonging to the current user."""
    category_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db_category_data = {
        "id": category_id,
        "user_id": current_user.id,
        "name": category_data.name,
        "description": category_data.description,
        "icon": category_data.icon,
        "created_at": now,
        "updated_at": now,
    }
    success = await db.create_category(db_category_data)
    if not success:
        logger.error(f"Failed to create secret category '{category_data.name}' for user {current_user.id}")
        raise HTTPException(status_code=500, detail="Could not create secret category.")

    logger.info(f"Created secret category '{category_data.name}' (ID: {category_id}) for user {current_user.id}")
    # Return the created data, conforming to the response model
    return SecretCategoryResponse(**db_category_data)


@router.get(
    "/categories",
    response_model=List[SecretCategoryResponse],
    summary="List user's secret categories"
)
async def list_secret_categories(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: DatabaseManager = Depends(get_db_manager)
) -> List[SecretCategoryResponse]:
    """Retrieves all secret categories belonging to the current user."""
    categories_data = await db.get_user_categories(current_user.id)
    return [SecretCategoryResponse(**cat) for cat in categories_data]


@router.delete(
    "/categories/{category_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a secret category"
)
async def delete_secret_category(
    category_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: DatabaseManager = Depends(get_db_manager)
):
    """
    Deletes a secret category belonging to the current user.
    Note: Does not automatically delete secrets within the category.
    """
    # Verify ownership first
    category = await db.get_category(current_user.id, category_id)
    if not category:
        logger.warning(f"Attempt to delete non-existent or unauthorized category {category_id} by user {current_user.id}")
        raise NotFoundException(detail="Secret category not found or access denied.")

    # TODO: Decide on behavior for secrets in the category.
    # Option 1: Prevent deletion if category contains secrets.
    # Option 2: Delete secrets (potentially dangerous).
    # Option 3: Orphan secrets (remove category_id).
    # For now, just delete the category itself.
    secrets_in_category = await db.get_user_secrets(user_id=current_user.id, category_id=category_id)
    if secrets_in_category:
         logger.warning(f"Attempt to delete category {category_id} which contains secrets. Deletion prevented.")
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Cannot delete category: contains secrets.")


    success = await db.delete_category(category_id)
    if not success:
        logger.error(f"Failed to delete secret category {category_id} for user {current_user.id}")
        raise HTTPException(status_code=500, detail="Could not delete secret category.")

    logger.info(f"Deleted secret category {category_id} for user {current_user.id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Secret Endpoints ---

@router.post(
    "/",
    response_model=SecretResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new secret"
)
async def create_secret(
    secret_data: SecretCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: DatabaseManager = Depends(get_db_manager)
) -> SecretResponse:
    """Creates a new encrypted secret belonging to the current user."""
    # Verify category exists and belongs to user
    category = await db.get_category(current_user.id, secret_data.category_id)
    if not category:
        logger.warning(f"Attempt to create secret in invalid category {secret_data.category_id} by user {current_user.id}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid category ID.")

    secret_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    user_key = await get_user_key_placeholder(current_user) # Get user's key (placeholder)
    encrypted_value = encrypt_value(secret_data.value, user_key)

    db_secret_data = {
        "id": secret_id,
        "user_id": current_user.id,
        "category_id": secret_data.category_id,
        "name": secret_data.name,
        "encrypted_value": encrypted_value, # Store encrypted value
        "notes": secret_data.notes,
        "tags": list(set(secret_data.tags)), # Ensure unique tags
        "created_at": now,
        "updated_at": now,
    }

    success = await db.create_secret(db_secret_data)
    if not success:
        logger.error(f"Failed to create secret '{secret_data.name}' for user {current_user.id}")
        raise HTTPException(status_code=500, detail="Could not create secret.")

    logger.info(f"Created secret '{secret_data.name}' (ID: {secret_id}) for user {current_user.id}")

    # Return data conforming to SecretResponse (no value)
    response_data = db_secret_data.copy()
    del response_data["encrypted_value"] # Remove encrypted value from response
    return SecretResponse(**response_data)


@router.get(
    "/",
    response_model=List[SecretResponse],
    summary="List user's secrets"
)
async def list_secrets(
    category_id: Optional[str] = Query(None, description="Filter secrets by category ID"),
    tag: Optional[str] = Query(None, description="Filter secrets by tag"),
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: DatabaseManager = Depends(get_db_manager)
) -> List[SecretResponse]:
    """Retrieves secrets belonging to the current user, optionally filtered."""
    secrets_data = await db.get_user_secrets(
        user_id=current_user.id,
        category_id=category_id,
        tag=tag
    )
    # Convert to response model (excluding encrypted value)
    response_list = []
    for sec in secrets_data:
        sec_copy = sec.copy()
        if "encrypted_value" in sec_copy:
            del sec_copy["encrypted_value"]
        response_list.append(SecretResponse(**sec_copy))
    return response_list


@router.get(
    "/{secret_id}/value",
    response_model=Dict[str, str], # Return just the decrypted value
    summary="Get the decrypted value of a specific secret"
)
async def get_secret_value(
    secret_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: DatabaseManager = Depends(get_db_manager)
) -> Dict[str, str]:
    """Retrieves and decrypts the value of a specific secret belonging to the user."""
    secret_data = await db.get_secret(secret_id)
    if not secret_data or secret_data.get("user_id") != current_user.id:
        logger.warning(f"Attempt to access value of non-existent or unauthorized secret {secret_id} by user {current_user.id}")
        raise NotFoundException(detail="Secret not found or access denied.")

    encrypted_value = secret_data.get("encrypted_value")
    if not encrypted_value:
         logger.error(f"Encrypted value missing for secret {secret_id}")
         raise HTTPException(status_code=500, detail="Secret data corrupted (missing value).")

    user_key = await get_user_key_placeholder(current_user)
    decrypted_value = decrypt_value(encrypted_value, user_key)

    logger.info(f"Decrypted value accessed for secret {secret_id} by user {current_user.id}")
    return {"value": decrypted_value}


@router.put(
    "/{secret_id}",
    response_model=SecretResponse,
    summary="Update an existing secret"
)
async def update_secret(
    secret_id: str,
    update_data: SecretUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: DatabaseManager = Depends(get_db_manager)
) -> SecretResponse:
    """Updates details of an existing secret belonging to the user."""
    # Verify secret exists and belongs to user
    existing_secret = await db.get_secret(secret_id)
    if not existing_secret or existing_secret.get("user_id") != current_user.id:
        logger.warning(f"Attempt to update non-existent or unauthorized secret {secret_id} by user {current_user.id}")
        raise NotFoundException(detail="Secret not found or access denied.")

    # Prepare updates for the database, handling potential encryption
    db_updates: Dict[str, Any] = {}
    has_changes = False

    if update_data.name is not None and update_data.name != existing_secret.get("name"):
        db_updates["name"] = update_data.name
        has_changes = True
    if update_data.notes is not None and update_data.notes != existing_secret.get("notes"):
        db_updates["notes"] = update_data.notes
        has_changes = True
    if update_data.tags is not None:
         new_tags = list(set(update_data.tags))
         if set(new_tags) != set(existing_secret.get("tags", [])):
              db_updates["tags"] = new_tags
              has_changes = True
    if update_data.category_id is not None and update_data.category_id != existing_secret.get("category_id"):
        # Verify the new category exists and belongs to the user
        new_category = await db.get_category(current_user.id, update_data.category_id)
        if not new_category:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid target category ID.")
        db_updates["category_id"] = update_data.category_id
        has_changes = True

    # Handle value update (encrypt new value)
    if update_data.value is not None:
        user_key = await get_user_key_placeholder(current_user)
        new_encrypted_value = encrypt_value(update_data.value, user_key)
        # Only update if the encrypted value actually changes
        if new_encrypted_value != existing_secret.get("encrypted_value"):
            db_updates["encrypted_value"] = new_encrypted_value
            has_changes = True

    if not has_changes:
         logger.info(f"No changes detected for secret {secret_id}. Update skipped.")
         # Return current state without hitting DB again
         response_data = existing_secret.copy()
         if "encrypted_value" in response_data: del response_data["encrypted_value"]
         return SecretResponse(**response_data)

    db_updates["updated_at"] = datetime.now(timezone.utc)

    success = await db.update_secret(secret_id, db_updates)
    if not success:
        logger.error(f"Failed to update secret {secret_id} for user {current_user.id}")
        raise HTTPException(status_code=500, detail="Could not update secret.")

    logger.info(f"Updated secret {secret_id} for user {current_user.id}")

    # Get updated data to return (or construct from updates)
    updated_secret_data = {**existing_secret, **db_updates}
    if "encrypted_value" in updated_secret_data: del updated_secret_data["encrypted_value"]
    return SecretResponse(**updated_secret_data)


@router.delete(
    "/{secret_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a secret"
)
async def delete_secret(
    secret_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: DatabaseManager = Depends(get_db_manager)
):
    """Deletes a specific secret belonging to the current user."""
    # Verify ownership
    secret = await db.get_secret(secret_id)
    if not secret or secret.get("user_id") != current_user.id:
        logger.warning(f"Attempt to delete non-existent or unauthorized secret {secret_id} by user {current_user.id}")
        raise NotFoundException(detail="Secret not found or access denied.")

    success = await db.delete_secret(secret_id)
    if not success:
        logger.error(f"Failed to delete secret {secret_id} for user {current_user.id}")
        raise HTTPException(status_code=500, detail="Could not delete secret.")

    logger.info(f"Deleted secret {secret_id} for user {current_user.id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)