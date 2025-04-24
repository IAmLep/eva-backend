"""
Secrets Management Router for EVA backend.

Provides endpoints for creating, retrieving, updating, and deleting
secret categories and individual secrets (like API keys, passwords, etc.).
Secrets are encrypted before storage.
"""

import logging
import uuid
from typing import Annotated, Any, Dict, List, Optional
# Import datetime and timezone for timestamping
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from pydantic import BaseModel, Field, field_validator

# --- Local Imports ---
from auth import get_current_active_user # Dependency for authentication
from database import get_db_manager, DatabaseManager # Database interaction
from models import User # User model
# Import encryption utilities (assuming they exist in utils.py or a dedicated crypto module)
from utils import encrypt_data, decrypt_data # Placeholder names (removed generate_key if not used)
from exceptions import DatabaseError, NotFoundException, AuthorizationError # Specific errors
# Assuming auth module might have key retrieval logic
import auth

# --- Router Setup ---
router = APIRouter(
    prefix="/secrets",
    tags=["Secrets Management"],
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Not found"},
        400: {"description": "Bad Request"},
        403: {"description": "Forbidden"}, # User doesn't own the resource
        500: {"description": "Internal Server Error"}
    }
)

# Logger configuration
logger = logging.getLogger(__name__)

# --- Pydantic Models for Secrets ---

class SecretCategoryBase(BaseModel):
    """Base model for secret category data."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    icon: Optional[str] = Field(None, max_length=50) # e.g., FontAwesome class or emoji

class SecretCategoryCreate(SecretCategoryBase):
    """Model for creating a new secret category."""
    pass # Inherits all fields from Base

class SecretCategoryResponse(SecretCategoryBase):
    """Model for returning secret category details."""
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True # Changed from orm_mode for Pydantic v2

class SecretBase(BaseModel):
    """Base model for secret data."""
    name: str = Field(..., min_length=1, max_length=100)
    username: Optional[str] = Field(None, max_length=200)
    # Secret value is handled separately due to encryption
    notes: Optional[str] = Field(None, max_length=2000)
    tags: List[str] = Field(default_factory=list)
    category_id: str # Link to the category

class SecretCreate(SecretBase):
    """Model for creating a new secret."""
    secret_value: str = Field(..., min_length=1) # The actual secret to encrypt

class SecretUpdate(BaseModel):
    """Model for updating an existing secret (allows partial updates)."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    username: Optional[str] = Field(None, max_length=200)
    secret_value: Optional[str] = Field(None, min_length=1) # If provided, will be re-encrypted
    notes: Optional[str] = Field(None, max_length=2000)
    tags: Optional[List[str]] = None
    category_id: Optional[str] = None # Allow moving secret between categories

class SecretResponse(SecretBase):
    """Model for returning secret details (EXCLUDES encrypted value)."""
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    # DO NOT return the encrypted_value or decrypted secret_value here

    class Config:
        from_attributes = True # Changed from orm_mode for Pydantic v2

class DecryptedSecretResponse(SecretResponse):
    """Model specifically for returning a decrypted secret value."""
    secret_value: str # The decrypted secret

# --- Secret Category Endpoints ---

@router.post(
    "/categories",
    response_model=SecretCategoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new secret category"
)
async def create_secret_category(
    category_data: SecretCategoryCreate, # No default (Body)
    current_user: Annotated[User, Depends(get_current_active_user)], # No default (Depends)
    db: DatabaseManager = Depends(get_db_manager) # Default (Depends)
) -> SecretCategoryResponse:
    """Creates a new category to organize secrets for the current user."""
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
    try:
        # Assuming db.create_category exists and takes the dict
        success = await db.create_category(db_category_data)
        if not success:
            raise DatabaseError("Failed to save category to database.")
    except DatabaseError as e:
        logger.error(f"DB Error creating category for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not create category.")
    except Exception as e:
        logger.exception(f"Unexpected error creating category for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


    logger.info(f"Created secret category '{category_data.name}' (ID: {category_id}) for user {current_user.id}")
    # Return the created data conforming to the response model
    return SecretCategoryResponse(**db_category_data)


@router.get(
    "/categories",
    response_model=List[SecretCategoryResponse],
    summary="List all secret categories for the user"
)
async def list_secret_categories(
    current_user: Annotated[User, Depends(get_current_active_user)], # No default
    db: DatabaseManager = Depends(get_db_manager) # Default
) -> List[SecretCategoryResponse]:
    """Retrieves all secret categories belonging to the current user."""
    try:
        # Assuming db.get_user_categories exists
        categories_data = await db.get_user_categories(current_user.id)
        # Convert list of dicts to list of response models
        return [SecretCategoryResponse(**cat) for cat in categories_data]
    except DatabaseError as e:
        logger.error(f"DB Error listing categories for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve categories.")
    except Exception as e:
        logger.exception(f"Unexpected error listing categories for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# CORRECTED: Reordered parameters
@router.get(
    "/categories/{category_id}",
    response_model=SecretCategoryResponse,
    summary="Get a specific secret category"
)
async def get_secret_category(
    current_user: Annotated[User, Depends(get_current_active_user)], # No default
    category_id: str = Path(..., description="The ID of the category to retrieve"), # Default (Path)
    db: DatabaseManager = Depends(get_db_manager) # Default (Depends)
) -> SecretCategoryResponse:
    """Retrieves details of a specific secret category by its ID."""
    try:
        # Assuming db.get_category exists
        category_data = await db.get_category(user_id=current_user.id, category_id=category_id)
        if not category_data:
            raise NotFoundException(f"Category {category_id} not found.")
        # Authorization implicitly handled by get_category checking user_id
        return SecretCategoryResponse(**category_data)
    except NotFoundException as e:
        logger.warning(f"Category {category_id} not found for user {current_user.id}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except DatabaseError as e:
        logger.error(f"DB Error getting category {category_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve category.")
    except Exception as e:
        logger.exception(f"Unexpected error getting category {category_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# CORRECTED: Reordered parameters
@router.put(
    "/categories/{category_id}",
    response_model=SecretCategoryResponse,
    summary="Update a secret category"
)
async def update_secret_category(
    update_data: SecretCategoryCreate, # No default (Body)
    current_user: Annotated[User, Depends(get_current_active_user)], # No default (Depends)
    category_id: str = Path(..., description="The ID of the category to update"), # Default (Path)
    db: DatabaseManager = Depends(get_db_manager) # Default (Depends)
) -> SecretCategoryResponse:
    """Updates the details of an existing secret category."""
    try:
        # 1. Verify category exists and belongs to user
        existing_category = await db.get_category(user_id=current_user.id, category_id=category_id)
        if not existing_category:
            raise NotFoundException(f"Category {category_id} not found.")

        # 2. Prepare update dictionary
        updates = update_data.model_dump(exclude_unset=True) # Get only provided fields
        updates["updated_at"] = datetime.now(timezone.utc)

        # 3. Perform update
        # Assuming db.update_category exists and takes category_id and updates dict
        success = await db.update_category(category_id=category_id, updates=updates)
        if not success:
            raise DatabaseError("Failed to update category in database.")

        # 4. Fetch updated data to return
        updated_category_data = await db.get_category(user_id=current_user.id, category_id=category_id)
        if not updated_category_data: # Should not happen if update succeeded
            raise DatabaseError("Failed to retrieve category after update.")

        logger.info(f"Updated secret category {category_id} for user {current_user.id}")
        return SecretCategoryResponse(**updated_category_data)

    except NotFoundException as e:
        logger.warning(f"Update failed: Category {category_id} not found for user {current_user.id}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except DatabaseError as e:
        logger.error(f"DB Error updating category {category_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not update category.")
    except Exception as e:
        logger.exception(f"Unexpected error updating category {category_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# CORRECTED: Reordered parameters
@router.delete(
    "/categories/{category_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a secret category"
)
async def delete_secret_category(
    current_user: Annotated[User, Depends(get_current_active_user)], # No default (Depends)
    category_id: str = Path(..., description="The ID of the category to delete"), # Default (Path)
    db: DatabaseManager = Depends(get_db_manager) # Default (Depends)
):
    """
    Deletes a secret category.
    WARNING: This might orphan secrets if not handled carefully.
    Consider preventing deletion if category contains secrets, or reassigning them.
    """
    try:
        # 1. Verify category exists and belongs to user
        existing_category = await db.get_category(user_id=current_user.id, category_id=category_id)
        if not existing_category:
            raise NotFoundException(f"Category {category_id} not found.")

        # TODO: Add check here - prevent deletion if secrets exist in this category?
        # secrets_in_category = await db.get_user_secrets(user_id=current_user.id, category_id=category_id)
        # if secrets_in_category:
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete category: contains secrets.")

        # 2. Perform deletion
        # Assuming db.delete_category exists
        success = await db.delete_category(category_id=category_id)
        if not success:
            # This might happen if the category was deleted between check and delete, or DB error
            raise DatabaseError("Failed to delete category from database.")

        logger.info(f"Deleted secret category {category_id} for user {current_user.id}")
        # No content to return for 204

    except NotFoundException:
        # If already not found, treat as success (idempotent delete)
        logger.warning(f"Deletion skipped: Category {category_id} not found for user {current_user.id}.")
        # Return 204 even if not found
    except DatabaseError as e:
        logger.error(f"DB Error deleting category {category_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not delete category.")
    except Exception as e:
        logger.exception(f"Unexpected error deleting category {category_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    # Return None for 204 response implicitly


# --- Secret Endpoints ---

@router.post(
    "/",
    response_model=SecretResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new secret"
)
async def create_secret(
    secret_data: SecretCreate, # No default (Body)
    current_user: Annotated[User, Depends(get_current_active_user)], # No default (Depends)
    db: DatabaseManager = Depends(get_db_manager) # Default (Depends)
) -> SecretResponse:
    """Creates a new secret, encrypting the secret value before storage."""
    # 1. Verify category exists and belongs to user
    try:
        category = await db.get_category(user_id=current_user.id, category_id=secret_data.category_id)
        if not category:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Category {secret_data.category_id} not found or invalid.")
    except DatabaseError as e:
         logger.error(f"DB error checking category {secret_data.category_id} for user {current_user.id}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Could not verify category.")
    except Exception as e:
        logger.exception(f"Unexpected error checking category {secret_data.category_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


    # 2. Encrypt the secret value
    try:
        # Placeholder: Derive or fetch encryption key for the user
        encryption_key = await auth.get_user_encryption_key(current_user.id) # Needs implementation in auth.py
        encrypted_value = encrypt_data(secret_data.secret_value.encode(), encryption_key)
    except NotImplementedError:
         logger.error("Encryption key retrieval not implemented.")
         raise HTTPException(status_code=501, detail="Secret encryption mechanism not available.")
    except Exception as e:
        logger.error(f"Encryption failed for new secret for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to encrypt secret value.")

    # 3. Prepare data for DB
    secret_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db_secret_data = {
        "id": secret_id,
        "user_id": current_user.id,
        "category_id": secret_data.category_id,
        "name": secret_data.name,
        "username": secret_data.username,
        "encrypted_value": encrypted_value, # Store encrypted value
        "notes": secret_data.notes,
        "tags": secret_data.tags,
        "created_at": now,
        "updated_at": now,
    }

    # 4. Save to DB
    try:
        # Assuming db.create_secret exists
        success = await db.create_secret(db_secret_data)
        if not success:
            raise DatabaseError("Failed to save secret to database.")
    except DatabaseError as e:
        logger.error(f"DB Error creating secret for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not create secret.")
    except Exception as e:
        logger.exception(f"Unexpected error creating secret for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


    logger.info(f"Created secret '{secret_data.name}' (ID: {secret_id}) for user {current_user.id}")

    # 5. Return response (excluding sensitive encrypted value)
    response_data = db_secret_data.copy()
    del response_data["encrypted_value"] # Ensure encrypted value is not returned

    return SecretResponse(**response_data)


@router.get(
    "/",
    response_model=List[SecretResponse],
    summary="List all secrets for the user"
)
async def list_secrets(
    current_user: Annotated[User, Depends(get_current_active_user)], # No default
    db: DatabaseManager = Depends(get_db_manager),                  # Default
    category_id: Optional[str] = Query(None, description="Filter secrets by category ID"), # Default
    tag: Optional[str] = Query(None, description="Filter secrets by tag") # Default
) -> List[SecretResponse]:
    """Retrieves all secrets belonging to the current user, optionally filtered."""
    try:
        # Assuming db.get_user_secrets exists
        secrets_data = await db.get_user_secrets(user_id=current_user.id, category_id=category_id, tag=tag)
        # Convert list of dicts, ensuring encrypted value is excluded
        response_list = []
        for sec in secrets_data:
             response_data = sec.copy()
             response_data.pop("encrypted_value", None) # Remove encrypted value if present
             response_list.append(SecretResponse(**response_data))
        return response_list
    except DatabaseError as e:
        logger.error(f"DB Error listing secrets for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve secrets.")
    except Exception as e:
        logger.exception(f"Unexpected error listing secrets for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# CORRECTED: Reordered parameters
@router.get(
    "/{secret_id}/decrypt",
    response_model=DecryptedSecretResponse,
    summary="Get and decrypt a specific secret"
)
async def get_decrypted_secret(
    current_user: Annotated[User, Depends(get_current_active_user)], # No default
    secret_id: str = Path(..., description="The ID of the secret to retrieve and decrypt"), # Default (Path)
    db: DatabaseManager = Depends(get_db_manager) # Default (Depends)
) -> DecryptedSecretResponse:
    """Retrieves and decrypts a specific secret by its ID."""
    try:
        # 1. Get secret data from DB
        # Assuming db.get_secret returns the dict including encrypted_value
        secret_data = await db.get_secret(user_id=current_user.id, secret_id=secret_id)
        if not secret_data:
            raise NotFoundException(f"Secret {secret_id} not found.")
        # Authorization implicitly handled by db.get_secret checking user_id

        # 3. Decrypt the value
        encrypted_value = secret_data.get("encrypted_value")
        if not encrypted_value:
             logger.error(f"Secret {secret_id} found but missing encrypted value.")
             raise HTTPException(status_code=500, detail="Secret data is corrupted.")

        try:
            # Placeholder: Derive or fetch encryption key
            encryption_key = await auth.get_user_encryption_key(current_user.id) # Needs implementation
            decrypted_value = decrypt_data(encrypted_value, encryption_key).decode()
        except NotImplementedError:
             logger.error("Encryption key retrieval not implemented.")
             raise HTTPException(status_code=501, detail="Secret decryption mechanism not available.")
        except Exception as e:
            logger.error(f"Decryption failed for secret {secret_id} for user {current_user.id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to decrypt secret value. Key might be incorrect or data corrupted.")

        # 4. Prepare and return response
        response_data = secret_data.copy()
        response_data.pop("encrypted_value", None) # Remove encrypted value
        response_data["secret_value"] = decrypted_value # Add decrypted value

        return DecryptedSecretResponse(**response_data)

    except NotFoundException as e:
        logger.warning(f"Decryption failed: Secret {secret_id} not found for user {current_user.id}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    # AuthorizationError check removed as db.get_secret handles it
    except DatabaseError as e:
        logger.error(f"DB Error getting secret {secret_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve secret.")
    except Exception as e:
        logger.exception(f"Unexpected error getting/decrypting secret {secret_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# CORRECTED: Reordered parameters
@router.put(
    "/{secret_id}",
    response_model=SecretResponse,
    summary="Update a secret"
)
async def update_secret(
    update_data: SecretUpdate, # No default (Body)
    current_user: Annotated[User, Depends(get_current_active_user)], # No default (Depends)
    secret_id: str = Path(..., description="The ID of the secret to update"), # Default (Path)
    db: DatabaseManager = Depends(get_db_manager) # Default (Depends)
) -> SecretResponse:
    """Updates an existing secret. Re-encrypts the value if provided."""
    try:
        # 1. Verify secret exists and belongs to user
        # Assuming db.get_secret checks ownership or we do it manually after fetch
        existing_secret = await db.get_secret(user_id=current_user.id, secret_id=secret_id)
        if not existing_secret:
            raise NotFoundException(f"Secret {secret_id} not found.")
        # Authorization implicitly handled by db.get_secret

        # 2. Prepare update dictionary
        updates = update_data.model_dump(exclude_unset=True)

        # 3. Handle re-encryption if secret_value is provided
        if "secret_value" in updates and updates["secret_value"] is not None:
            new_secret_value = updates.pop("secret_value") # Remove plain text value
            try:
                # Placeholder: Derive or fetch encryption key
                encryption_key = await auth.get_user_encryption_key(current_user.id) # Needs implementation
                updates["encrypted_value"] = encrypt_data(new_secret_value.encode(), encryption_key)
            except NotImplementedError:
                 logger.error("Encryption key retrieval not implemented.")
                 raise HTTPException(status_code=501, detail="Secret encryption mechanism not available.")
            except Exception as e:
                logger.error(f"Encryption failed during secret update for {secret_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to encrypt new secret value.")
        elif "secret_value" in updates:
             del updates["secret_value"] # Remove if None

        # 4. Update timestamp
        updates["updated_at"] = datetime.now(timezone.utc)

        # 5. Perform update in DB
        # Assuming db.update_secret exists
        success = await db.update_secret(user_id=current_user.id, secret_id=secret_id, updates=updates)
        if not success:
            raise DatabaseError("Failed to update secret in database.")

        # 6. Fetch updated data to return
        updated_secret_data = await db.get_secret(user_id=current_user.id, secret_id=secret_id)
        if not updated_secret_data:
            raise DatabaseError("Failed to retrieve secret after update.")

        logger.info(f"Updated secret {secret_id} for user {current_user.id}")

        # 7. Prepare and return response (excluding encrypted value)
        response_data = updated_secret_data.copy()
        response_data.pop("encrypted_value", None)
        return SecretResponse(**response_data)

    except NotFoundException as e:
        logger.warning(f"Update failed: Secret {secret_id} not found for user {current_user.id}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    # AuthorizationError check removed as db methods handle it
    except DatabaseError as e:
        logger.error(f"DB Error updating secret {secret_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not update secret.")
    except Exception as e:
        logger.exception(f"Unexpected error updating secret {secret_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# CORRECTED: Reordered parameters
@router.delete(
    "/{secret_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a secret"
)
async def delete_secret(
    current_user: Annotated[User, Depends(get_current_active_user)], # No default
    secret_id: str = Path(..., description="The ID of the secret to delete"), # Default (Path)
    db: DatabaseManager = Depends(get_db_manager) # Default (Depends)
):
    """Deletes a specific secret by its ID."""
    try:
        # 1. Verify secret exists and belongs to user (implicitly via db method)
        # Assuming db.delete_secret handles NotFound and Authorization
        success = await db.delete_secret(user_id=current_user.id, secret_id=secret_id)
        if not success:
             # This could mean not found or db error, db method should raise appropriately
             # If db.delete_secret returns False on not found, handle it here:
             logger.warning(f"Deletion failed or skipped: Secret {secret_id} for user {current_user.id}.")
             # We might still return 204 if not found is acceptable for idempotency
             # Or raise 404 if the DB method didn't already
             # raise NotFoundException(f"Secret {secret_id} not found.")

        logger.info(f"Deleted secret {secret_id} for user {current_user.id}")
        # No content for 204

    except NotFoundException: # Catch if db.delete_secret raises this
        logger.warning(f"Deletion skipped: Secret {secret_id} not found for user {current_user.id}.")
        # Return 204 even if not found (idempotent)
    except AuthorizationError as e: # Catch if db.delete_secret raises this
        logger.warning(f"Authorization error deleting secret {secret_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except DatabaseError as e:
        logger.error(f"DB Error deleting secret {secret_id} for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not delete secret.")
    except Exception as e:
        logger.exception(f"Unexpected error deleting secret {secret_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    # Return None for 204 response implicitly