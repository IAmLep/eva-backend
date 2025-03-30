import os
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Flag to determine if we're running in a Cloud environment
IS_CLOUD_ENVIRONMENT = os.environ.get("CLOUD_ENVIRONMENT", "false").lower() == "true"

# Development secrets - only used if IS_CLOUD_ENVIRONMENT is False
DEV_SECRETS = {
    "gemini-api-key": os.environ.get("GEMINI_API_KEY", ""),
    "openai-api-key": os.environ.get("OPENAI_API_KEY", ""),
    # Add other development secrets as needed
}

# Cache for secrets to avoid repeated API calls
SECRET_CACHE: Dict[str, str] = {}

async def get_secret(name: str) -> str:
    """
    Get a secret from the secret manager or environment.
    
    Args:
        name: Name of the secret
        
    Returns:
        Secret value
    """
    # Check cache first
    if name in SECRET_CACHE:
        return SECRET_CACHE[name]
    
    if IS_CLOUD_ENVIRONMENT:
        try:
            # Cloud environment - use Secret Manager
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            
            if not project_id:
                logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Cloud configuration error"
                )
            
            secret_path = f"projects/{project_id}/secrets/{name}/versions/latest"
            response = client.access_secret_version(request={"name": secret_path})
            secret_value = response.payload.data.decode("UTF-8")
            
            # Cache the secret
            SECRET_CACHE[name] = secret_value
            
            return secret_value
        
        except ImportError:
            logger.error("google-cloud-secret-manager package not installed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Secret manager not available"
            )
        
        except Exception as e:
            logger.error(f"Error retrieving secret '{name}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve secret"
            )
    else:
        # Development environment - use local secrets
        if name in DEV_SECRETS:
            # Cache the secret
            SECRET_CACHE[name] = DEV_SECRETS[name]
            return DEV_SECRETS[name]
        
        logger.error(f"Development secret '{name}' not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Secret '{name}' not found"
        )

async def list_available_secrets() -> list:
    """
    List all available secret names.
    
    Returns:
        List of secret names
    """
    if IS_CLOUD_ENVIRONMENT:
        try:
            # Cloud environment - use Secret Manager
            from google.cloud import secretmanager
            
            client = secretmanager.SecretManagerServiceClient()
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            
            if not project_id:
                logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Cloud configuration error"
                )
            
            parent = f"projects/{project_id}"
            secrets = client.list_secrets(request={"parent": parent})
            
            # Extract just the secret names for safety
            secret_names = []
            for secret in secrets:
                # Get just the final part of the name
                name = secret.name.split('/')[-1]
                secret_names.append(name)
            
            return secret_names
        
        except ImportError:
            logger.error("google-cloud-secret-manager package not installed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Secret manager not available"
            )
        
        except Exception as e:
            logger.error(f"Error listing secrets: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list secrets"
            )
    else:
        # Development environment - return keys from DEV_SECRETS
        return list(DEV_SECRETS.keys())