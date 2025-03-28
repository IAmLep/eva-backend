from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from auth import DeviceValidationResponse, validate_device_token
from secret_manager import get_secret, list_available_secrets

router = APIRouter(prefix="/secrets", tags=["secrets"])

class SecretResponse(BaseModel):
    """Response model for secret values."""
    name: str
    value: str

class SecretListResponse(BaseModel):
    """Response model for listing available secrets."""
    secrets: List[str]

async def validate_device_token_dependency(token: str) -> DeviceValidationResponse:
    """Dependency to validate device token."""
    validation = validate_device_token(token)
    if not validation.valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid device token"
        )
    return validation

@router.get("/{secret_name}", response_model=SecretResponse)
async def get_secret_endpoint(
    secret_name: str,
    validation: DeviceValidationResponse = Depends(validate_device_token_dependency)
):
    """
    Get a secret by name.
    Requires a valid device token.
    """
    secret_value = await get_secret(secret_name)
    return SecretResponse(name=secret_name, value=secret_value)

@router.get("/", response_model=SecretListResponse)
async def list_secrets_endpoint(
    validation: DeviceValidationResponse = Depends(validate_device_token_dependency)
):
    """
    List all available secrets.
    Requires a valid device token.
    """
    secret_names = await list_available_secrets()
    return SecretListResponse(secrets=secret_names)