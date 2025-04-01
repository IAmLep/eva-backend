"""
Secret Manager module for EVA backend.

This module provides secure storage and retrieval of sensitive information
using Google Cloud Secret Manager with local caching.

Last updated: 2025-04-01 10:50:57
Version: v1.8.6
Created by: IAmLep
"""

import base64
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from google.cloud import secretmanager
from google.cloud.exceptions import NotFound

from cache_manager import cached
from config import get_settings
from exceptions import ConfigurationError, NotFoundException

# Logger configuration
logger = logging.getLogger(__name__)


class SecretManager:
    """
    Secret Manager for secure handling of sensitive information.
    
    This class provides an interface for storing and retrieving secrets
    using Google Cloud Secret Manager with local caching for performance.
    """
    
    def __init__(self):
        """Initialize Secret Manager with settings."""
        self.settings = get_settings()
        self.project_id = self.settings.GOOGLE_CLOUD_PROJECT
        
        if not self.project_id:
            logger.warning("No Google Cloud project ID configured, using local secrets only")
            self.client = None
        else:
            try:
                self.client = secretmanager.SecretManagerServiceClient()
                logger.info("Secret Manager client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Secret Manager client: {str(e)}")
                self.client = None
        
        # Initialize encryption key
        self.encryption_key = self._get_encryption_key()
    
    def _get_encryption_key(self) -> bytes:
        """
        Get or generate encryption key.
        
        Returns:
            bytes: Encryption key
            
        Raises:
            ConfigurationError: If encryption key cannot be securely initialized
        """
        try:
            # Try to get key from environment
            env_key = os.environ.get("SECRET_ENCRYPTION_KEY")
            if env_key:
                # Ensure key is valid base64
                try:
                    key = base64.urlsafe_b64decode(env_key)
                    if len(key) == 32:  # Valid Fernet key
                        return key
                except Exception:
                    logger.warning("Invalid encryption key in environment, generating new key")
            
            # Try to get from settings
            settings_key = getattr(self.settings, "SECRET_ENCRYPTION_KEY", None)
            if settings_key:
                try:
                    key = base64.urlsafe_b64decode(settings_key)
                    if len(key) == 32:  # Valid Fernet key
                        return key
                except Exception:
                    logger.warning("Invalid encryption key in settings, generating new key")
            
            # Generate key from app secret
            if self.settings.SECRET_KEY:
                salt = b"eva_secret_manager_salt"
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(self.settings.SECRET_KEY.encode()))
                return base64.urlsafe_b64decode(key)
            
            # Generate random key as last resort
            key = Fernet.generate_key()
            logger.warning("Using randomly generated encryption key - secrets will be lost on restart")
            return base64.urlsafe_b64decode(key)
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption key: {str(e)}")
            raise ConfigurationError("Failed to initialize encryption key")
    
    def _encrypt(self, data: str) -> str:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            str: Encrypted data as base64
        """
        f = Fernet(base64.urlsafe_b64encode(self.encryption_key))
        encrypted = f.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            str: Decrypted data
            
        Raises:
            ValueError: If decryption fails
        """
        try:
            f = Fernet(base64.urlsafe_b64encode(self.encryption_key))
            encrypted = base64.b64decode(encrypted_data)
            decrypted = f.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {str(e)}")
            raise ValueError("Failed to decrypt data")
    
    async def create_secret(
        self, 
        secret_id: str, 
        secret_value: str,
        user_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a new secret.
        
        Args:
            secret_id: Secret identifier
            secret_value: Secret value
            user_id: Optional user ID for user-specific secrets
            labels: Optional labels for the secret
            
        Returns:
            str: Full secret name
            
        Raises:
            ConfigurationError: If Secret Manager is not configured
        """
        # Prefix user-specific secrets
        if user_id:
            secret_id = f"user-{user_id}-{secret_id}"
        
        # Create secret in Google Cloud Secret Manager if available
        if self.client and self.project_id:
            try:
                # Build the resource name
                parent = f"projects/{self.project_id}"
                
                # Create the secret
                secret = self.client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": {
                            "replication": {"automatic": {}},
                            "labels": labels or {},
                        },
                    }
                )
                
                # Add the secret version
                secret_version = self.client.add_secret_version(
                    request={
                        "parent": secret.name,
                        "payload": {"data": secret_value.encode()},
                    }
                )
                
                logger.info(f"Created secret: {secret.name}")
                return secret.name
            
            except Exception as e:
                logger.error(f"Failed to create secret in Google Cloud: {str(e)}")
                # Fall back to local storage
        
        # Store locally (encrypted)
        try:
            # Encrypt the secret value
            encrypted_value = self._encrypt(secret_value)
            
            # Store in local secret cache
            local_secrets = self._get_local_secrets()
            local_secrets[secret_id] = {
                "value": encrypted_value,
                "user_id": user_id,
                "labels": labels or {},
            }
            self._save_local_secrets(local_secrets)
            
            logger.info(f"Created local secret: {secret_id}")
            return f"local/{secret_id}"
        
        except Exception as e:
            logger.error(f"Failed to create local secret: {str(e)}")
            raise ConfigurationError(f"Failed to create secret: {str(e)}")
    
    @cached(ttl=300, key_prefix="secret")
    async def get_secret(
        self, 
        secret_id: str,
        user_id: Optional[str] = None,
        version: str = "latest"
    ) -> str:
        """
        Get a secret value.
        
        Args:
            secret_id: Secret identifier
            user_id: Optional user ID for user-specific secrets
            version: Secret version
            
        Returns:
            str: Secret value
            
        Raises:
            NotFoundException: If secret not found
        """
        # Prefix user-specific secrets
        if user_id:
            secret_id = f"user-{user_id}-{secret_id}"
        
        # Try to get from Google Cloud Secret Manager if available
        if self.client and self.project_id:
            try:
                # Build the resource name
                name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
                
                # Get the secret
                response = self.client.access_secret_version(request={"name": name})
                
                # Return the secret value
                logger.info(f"Retrieved secret from Google Cloud: {secret_id}")
                return response.payload.data.decode()
            
            except NotFound:
                logger.warning(f"Secret not found in Google Cloud: {secret_id}")
                # Fall back to local storage
            
            except Exception as e:
                logger.error(f"Failed to get secret from Google Cloud: {str(e)}")
                # Fall back to local storage
        
        # Try to get from local storage
        try:
            local_secrets = self._get_local_secrets()
            
            if secret_id not in local_secrets:
                logger.warning(f"Secret not found locally: {secret_id}")
                raise NotFoundException(f"Secret not found: {secret_id}")
            
            # Decrypt the secret value
            encrypted_value = local_secrets[secret_id]["value"]
            secret_value = self._decrypt(encrypted_value)
            
            logger.info(f"Retrieved secret from local storage: {secret_id}")
            return secret_value
        
        except NotFoundException:
            # Re-raise not found exception
            raise
        
        except Exception as e:
            logger.error(f"Failed to get local secret: {str(e)}")
            raise NotFoundException(f"Secret not found or access failed: {secret_id}")
    
    async def update_secret(
        self, 
        secret_id: str, 
        secret_value: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update an existing secret.
        
        Args:
            secret_id: Secret identifier
            secret_value: New secret value
            user_id: Optional user ID for user-specific secrets
            
        Returns:
            bool: True if successful
            
        Raises:
            NotFoundException: If secret not found
        """
        # Prefix user-specific secrets
        if user_id:
            secret_id = f"user-{user_id}-{secret_id}"
        
        # Update in Google Cloud Secret Manager if available
        if self.client and self.project_id:
            try:
                # Check if secret exists
                name = f"projects/{self.project_id}/secrets/{secret_id}"
                self.client.get_secret(request={"name": name})
                
                # Add new secret version
                self.client.add_secret_version(
                    request={
                        "parent": name,
                        "payload": {"data": secret_value.encode()},
                    }
                )
                
                logger.info(f"Updated secret in Google Cloud: {secret_id}")
                return True
            
            except NotFound:
                logger.warning(f"Secret not found in Google Cloud for update: {secret_id}")
                # Fall back to local storage
            
            except Exception as e:
                logger.error(f"Failed to update secret in Google Cloud: {str(e)}")
                # Fall back to local storage
        
        # Update in local storage
        try:
            local_secrets = self._get_local_secrets()
            
            if secret_id not in local_secrets:
                logger.warning(f"Secret not found locally for update: {secret_id}")
                raise NotFoundException(f"Secret not found: {secret_id}")
            
            # Encrypt the secret value
            encrypted_value = self._encrypt(secret_value)
            
            # Update in local secret cache
            local_secrets[secret_id]["value"] = encrypted_value
            self._save_local_secrets(local_secrets)
            
            logger.info(f"Updated local secret: {secret_id}")
            return True
        
        except NotFoundException:
            # Re-raise not found exception
            raise
        
        except Exception as e:
            logger.error(f"Failed to update local secret: {str(e)}")
            raise ConfigurationError(f"Failed to update secret: {str(e)}")
    
    async def delete_secret(
        self, 
        secret_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Delete a secret.
        
        Args:
            secret_id: Secret identifier
            user_id: Optional user ID for user-specific secrets
            
        Returns:
            bool: True if successful
            
        Raises:
            NotFoundException: If secret not found
        """
        # Prefix user-specific secrets
        if user_id:
            secret_id = f"user-{user_id}-{secret_id}"
        
        # Try to delete from Google Cloud Secret Manager if available
        gcp_deleted = False
        if self.client and self.project_id:
            try:
                # Build the resource name
                name = f"projects/{self.project_id}/secrets/{secret_id}"
                
                # Delete the secret
                self.client.delete_secret(request={"name": name})
                
                logger.info(f"Deleted secret from Google Cloud: {secret_id}")
                gcp_deleted = True
            
            except NotFound:
                logger.warning(f"Secret not found in Google Cloud for deletion: {secret_id}")
                # Continue to check local storage
            
            except Exception as e:
                logger.error(f"Failed to delete secret from Google Cloud: {str(e)}")
                # Continue to check local storage
        
        # Try to delete from local storage
        try:
            local_secrets = self._get_local_secrets()
            
            if secret_id in local_secrets:
                # Remove from local secret cache
                del local_secrets[secret_id]
                self._save_local_secrets(local_secrets)
                
                logger.info(f"Deleted local secret: {secret_id}")
                return True
            
            # If we've already deleted from GCP, return success
            if gcp_deleted:
                return True
            
            # If neither found, raise exception
            logger.warning(f"Secret not found for deletion: {secret_id}")
            raise NotFoundException(f"Secret not found: {secret_id}")
        
        except NotFoundException:
            # Re-raise not found exception
            raise
        
        except Exception as e:
            logger.error(f"Failed to delete local secret: {str(e)}")
            raise ConfigurationError(f"Failed to delete secret: {str(e)}")
    
    async def list_secrets(
        self, 
        user_id: Optional[str] = None,
        filter_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available secrets.
        
        Args:
            user_id: Optional user ID to filter user-specific secrets
            filter_prefix: Optional prefix to filter secret IDs
            
        Returns:
            List[Dict[str, Any]]: List of secret information
        """
        result = []
        
        # List from Google Cloud Secret Manager if available
        if self.client and self.project_id:
            try:
                # Build the resource name
                parent = f"projects/{self.project_id}"
                
                # List secrets
                filter_expr = ""
                if user_id:
                    filter_expr = f'labels.user_id="{user_id}"'
                
                response = self.client.list_secrets(
                    request={"parent": parent, "filter": filter_expr}
                )
                
                # Process secrets
                for secret in response:
                    secret_id = secret.name.split("/")[-1]
                    
                    # Filter by prefix if specified
                    if filter_prefix and not secret_id.startswith(filter_prefix):
                        continue
                    
                    # Filter by user if specified
                    if user_id and not secret_id.startswith(f"user-{user_id}-"):
                        continue
                    
                    # Add to result
                    result.append({
                        "id": secret_id,
                        "name": secret.name,
                        "create_time": secret.create_time.isoformat(),
                        "labels": dict(secret.labels),
                        "source": "google_cloud",
                    })
                
                logger.info(f"Listed {len(result)} secrets from Google Cloud")
            
            except Exception as e:
                logger.error(f"Failed to list secrets from Google Cloud: {str(e)}")
                # Continue to check local storage
        
        # List from local storage
        try:
            local_secrets = self._get_local_secrets()
            
            for secret_id, secret_data in local_secrets.items():
                # Filter by prefix if specified
                if filter_prefix and not secret_id.startswith(filter_prefix):
                    continue
                
                # Filter by user if specified
                if user_id:
                    if not secret_id.startswith(f"user-{user_id}-"):
                        if secret_data.get("user_id") != user_id:
                            continue
                
                # Add to result if not already present
                if not any(s["id"] == secret_id for s in result):
                    result.append({
                        "id": secret_id,
                        "name": f"local/{secret_id}",
                        "create_time": None,
                        "labels": secret_data.get("labels", {}),
                        "source": "local",
                    })
            
            logger.info(f"Listed {len(result)} total secrets")
            return result
        
        except Exception as e:
            logger.error(f"Failed to list local secrets: {str(e)}")
            return result
    
    def _get_local_secrets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get local secrets from file.
        
        Returns:
            Dict[str, Dict[str, Any]]: Local secrets
        """
        # Define path for local secrets
        secrets_file = os.path.join(os.path.dirname(__file__), ".local_secrets.json")
        
        # Try to read existing file
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read local secrets file: {str(e)}")
        
        # Return empty dict if file doesn't exist or read failed
        return {}
    
    def _save_local_secrets(self, secrets: Dict[str, Dict[str, Any]]) -> None:
        """
        Save local secrets to file.
        
        Args:
            secrets: Secrets to save
        """
        # Define path for local secrets
        secrets_file = os.path.join(os.path.dirname(__file__), ".local_secrets.json")
        
        # Save to file
        try:
            with open(secrets_file, "w") as f:
                json.dump(secrets, f)
            
            # Set restrictive permissions
            os.chmod(secrets_file, 0o600)
        except Exception as e:
            logger.error(f"Failed to save local secrets file: {str(e)}")


# Singleton instance
_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """
    Get Secret Manager singleton instance.
    
    Returns:
        SecretManager: Secret Manager instance
    """
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager


async def get_secret(
    secret_id: str,
    user_id: Optional[str] = None,
    version: str = "latest"
) -> str:
    """
    Get a secret value (convenience function).
    
    Args:
        secret_id: Secret identifier
        user_id: Optional user ID for user-specific secrets
        version: Secret version
        
    Returns:
        str: Secret value
    """
    secret_manager = get_secret_manager()
    return await secret_manager.get_secret(secret_id, user_id, version)


async def create_secret(
    secret_id: str,
    secret_value: str,
    user_id: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None
) -> str:
    """
    Create a new secret (convenience function).
    
    Args:
        secret_id: Secret identifier
        secret_value: Secret value
        user_id: Optional user ID for user-specific secrets
        labels: Optional labels for the secret
        
    Returns:
        str: Full secret name
    """
    secret_manager = get_secret_manager()
    return await secret_manager.create_secret(secret_id, secret_value, user_id, labels)


async def update_secret(
    secret_id: str,
    secret_value: str,
    user_id: Optional[str] = None
) -> bool:
    """
    Update an existing secret (convenience function).
    
    Args:
        secret_id: Secret identifier
        secret_value: New secret value
        user_id: Optional user ID for user-specific secrets
        
    Returns:
        bool: True if successful
    """
    secret_manager = get_secret_manager()
    return await secret_manager.update_secret(secret_id, secret_value, user_id)


async def delete_secret(
    secret_id: str,
    user_id: Optional[str] = None
) -> bool:
    """
    Delete a secret (convenience function).
    
    Args:
        secret_id: Secret identifier
        user_id: Optional user ID for user-specific secrets
        
    Returns:
        bool: True if successful
    """
    secret_manager = get_secret_manager()
    return await secret_manager.delete_secret(secret_id, user_id)