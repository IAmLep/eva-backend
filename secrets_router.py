"""
API Router for managing user secrets and categories.

Provides endpoints for CRUD operations on secret categories and individual secrets,
ensuring user ownership and authentication. Uses Fernet symmetric encryption.
"""

import logging
import uuid
import base64
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, Response # Import Response
from pydantic import BaseModel, Field, field_validator
from cryptography.fernet import Fernet, InvalidToken # pip install cryptography

# --- Local Imports ---
from auth import get_current_active_user # Authentication dependency
from database import get_db_manager, DatabaseManager # Use the DB manager
from exceptions import DatabaseError, NotFoundException, AuthorizationError, BadRequestError # Custom exceptions
from models import User # User model
from config import get_settings # Import settings to get master key

# --- Router Setup ---
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)

# --- Pydantic Models for Secrets ---
# (Defined previously, assuming they are correct)
class SecretCategoryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=200)
    icon: Optional[str] = Field(None, max_length=50)

class SecretCategoryResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class SecretCreate(BaseModel):
    category_id: str
    name: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., description="The actual secret value (will be encrypted)")
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class SecretUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    value: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    category_id: Optional[str] = None

class SecretResponse(BaseModel):
    id: str
    user_id: str
    category_id: str
    name: str
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

# --- Encryption Setup ---
# Load the master key securely from settings
try:
    settings = get_settings()
    # Decode the base64 encoded key from settings
    MASTER_KEY_BYTES = base64.urlsafe_b64decode(settings.MASTER_ENCRYPTION_KEY)
    if len(MASTER_KEY_BYTES) != 32:
        raise ConfigurationError("MASTER_ENCRYPTION_KEY in settings must be a 32-byte key after base64 decoding.")
    fernet_instance = Fernet(settings.MASTER_ENCRYPTION_KEY.encode()) # Fernet expects base64 encoded key
    logger.info("Fernet encryption initialized successfully using MASTER_ENCRYPTION_KEY.")
except (ImportError, ConfigurationError, ValueError, Exception) as e:
    logger.critical(f"CRITICAL: Failed to initialize encryption. Secrets will not work. Error: {e}", exc_info=True)
    # Define dummy functions to allow startup but log critical errors on use
    def encrypt_value(value: str) -> str:
        logger.critical("Encryption is not configured properly. Cannot encrypt secret.")
        raise ConfigurationError("Encryption service unavailable.")
    def decrypt_value(encrypted_value: str) -> str:
        logger.critical("Encryption is not configured properly. Cannot decrypt secret.")
        raise ConfigurationError("Encryption service unavailable.")
else:
    # Define actual encryption/decryption functions using the initialized Fernet instance
    def encrypt_value(value: str) -> str:
        """Encrypts a string value using the master Fernet instance."""
        if not isinstance(value, str):
             raise TypeError("Value to encrypt must be a string.")
        try:
            encrypted_bytes = fernet_instance.encrypt(value.encode('utf-8'))
            # Return as string for JSON/DB storage (Fernet token is URL-safe base64)
            return encrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}", exc_info=True)
            raise ValueError("Encryption process failed.")

    def decrypt_value(encrypted_token: str) -> str:
        """Decrypts a Fernet token string using the master Fernet instance."""
        if not isinstance(encrypted_token, str):
             raise TypeError("Token to decrypt must be a string.")
        try:
            decrypted_bytes = fernet_instance.decrypt(encrypted_token.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except InvalidToken:
            logger.error("Decryption failed: Invalid token or key.")
            raise ValueError("Decryption failed: Invalid token or key.")
        except Exception as e:
            logger.error(f"Decryption failed: {e}", exc_info=True)
            raise ValueError("Decryption process failed.")

# --- Category Endpoints (No changes needed from previous version) ---

@router.post("/categories", ...)
async def create_secret_category(...): ... # Assume implementation is correct

@router.get("/categories", ...)
async def list_secret_categories(...): ... # Assume implementation is correct

@router.delete("/categories/{category_id}", ...)
async def delete_secret_category(...): ... # Assume implementation is correct

# --- Secret Endpoints (Updated with Real Encryption) ---

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
    category = await db.get_category(current_user.id, secret_data.category_id)
    if not category:
        raise BadRequestError(detail="Invalid category ID.")

    secret_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    try:
        encrypted_value = encrypt_value(secret_data.value) # Use real encryption
    except (ConfigurationError, ValueError) as e:
         logger.error(f"Encryption failed during secret creation for user {current_user.id}: {e}")
         raise HTTPException(status_code=500, detail=f"Could not encrypt secret: {e}")

    db_secret_data = {
        "id": secret_id,
        "user_id": current_user.id,
        "category_id": secret_data.category_id,
        "name": secret_data.name,
        "encrypted_value": encrypted_value, # Store encrypted value
        "notes": secret_data.notes,
        "tags": list(set(secret_data.tags)),
        "created_at": now,
        "updated_at": now,
    }

    success = await db.create_secret(db_secret_data)
    if not success:
        raise DatabaseError(detail="Could not create secret.")

    logger.info(f"Created secret '{secret_data.name}' (ID: {secret_id}) for user {current_user.id}")
    response_data = db_secret_data.copy()
    del response_data["encrypted_value"]
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
    response_list = []
    for sec in secrets_data:
        sec_copy = sec.copy()
        if "encrypted_value" in sec_copy:
            del sec_copy["encrypted_value"]
        try:
            # Ensure data conforms to response model before appending
            response_list.append(SecretResponse(**sec_copy))
        except Exception as e:
             logger.warning(f"Skipping secret {sec.get('id')} in list due to validation error: {e}")
    return response_list


@router.get(
    "/{secret_id}/value",
    response_model=Dict[str, str],
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
        raise NotFoundException(detail="Secret not found or access denied.")

    encrypted_value = secret_data.get("encrypted_value")
    if not encrypted_value:
         raise DatabaseError(detail="Secret data corrupted (missing value).")

    try:
        decrypted_value = decrypt_value(encrypted_value) # Use real decryption
    except (ConfigurationError, ValueError) as e:
         logger.error(f"Decryption failed for secret {secret_id} for user {current_user.id}: {e}")
         raise HTTPException(status_code=500, detail=f"Could not decrypt secret: {e}")

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
    existing_secret = await db.get_secret(secret_id)
    if not existing_secret or existing_secret.get("user_id") != current_user.id:
        raise NotFoundException(detail="Secret not found or access denied.")

    db_updates: Dict[str, Any] = {}
    has_changes = False

    # ... (logic for updating name, notes, tags, category_id - unchanged) ...

    # Handle value update (encrypt new value)
    if update_data.value is not None:
        try:
            new_encrypted_value = encrypt_value(update_data.value) # Use real encryption
            if new_encrypted_value != existing_secret.get("encrypted_value"):
                db_updates["encrypted_value"] = new_encrypted_value
                has_changes = True
        except (ConfigurationError, ValueError) as e:
             logger.error(f"Encryption failed during secret update for user {current_user.id}: {e}")
             raise HTTPException(status_code=500, detail=f"Could not encrypt secret value for update: {e}")

    if not has_changes:
         response_data = existing_secret.copy()
         if "encrypted_value" in response_data: del response_data["encrypted_value"]
         return SecretResponse(**response_data)

    db_updates["updated_at"] = datetime.now(timezone.utc)

    success = await db.update_secret(secret_id, db_updates)
    if not success:
        raise DatabaseError(detail="Could not update secret.")

    logger.info(f"Updated secret {secret_id} for user {current_user.id}")
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
    secret = await db.get_secret(secret_id)
    if not secret or secret.get("user_id") != current_user.id:
        raise NotFoundException(detail="Secret not found or access denied.")

    success = await db.delete_secret(secret_id)
    if not success:
        raise DatabaseError(detail="Could not delete secret.")

    logger.info(f"Deleted secret {secret_id} for user {current_user.id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)