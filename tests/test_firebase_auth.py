"""
Tests for the Firebase authentication module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from models import UserInDB, UserRole
from exceptions import AuthenticationError, DatabaseError


class TestVerifyFirebaseToken:
    """Tests for firebase token verification."""

    @pytest.mark.asyncio
    async def test_firebase_not_available(self):
        """Test that an error is raised when Firebase is not configured."""
        with patch("firebase_auth.FIREBASE_AUTH_AVAILABLE", False):
            from firebase_auth import verify_firebase_token
            with pytest.raises(AuthenticationError, match="not configured"):
                await verify_firebase_token("fake-token")

    @pytest.mark.asyncio
    async def test_valid_token(self):
        """Test successful token verification."""
        mock_decoded = {
            "uid": "firebase-uid-123",
            "email": "user@example.com",
            "name": "Test User",
        }

        with patch("firebase_auth.FIREBASE_AUTH_AVAILABLE", True), \
             patch("firebase_auth.firebase_auth_admin") as mock_admin:
            mock_admin.verify_id_token.return_value = mock_decoded

            from firebase_auth import verify_firebase_token
            result = await verify_firebase_token("valid-token")

            assert result["uid"] == "firebase-uid-123"
            assert result["email"] == "user@example.com"
            mock_admin.verify_id_token.assert_called_once_with("valid-token")

    @pytest.mark.asyncio
    async def test_expired_token(self):
        """Test that expired tokens raise AuthenticationError."""
        with patch("firebase_auth.FIREBASE_AUTH_AVAILABLE", True), \
             patch("firebase_auth.firebase_auth_admin") as mock_admin:
            mock_admin.ExpiredIdTokenError = type("ExpiredIdTokenError", (Exception,), {})
            mock_admin.verify_id_token.side_effect = mock_admin.ExpiredIdTokenError("expired")

            from firebase_auth import verify_firebase_token
            with pytest.raises(AuthenticationError, match="expired"):
                await verify_firebase_token("expired-token")


class TestGetOrCreateFirebaseUser:
    """Tests for user creation/retrieval from Firebase tokens."""

    @pytest.fixture
    def decoded_token(self):
        return {
            "uid": "firebase-uid-123",
            "email": "newuser@example.com",
            "name": "New User",
            "picture": "https://example.com/photo.jpg",
        }

    @pytest.fixture
    def existing_user(self):
        return UserInDB(
            id="existing-user-id",
            username="existinguser",
            email="newuser@example.com",
            full_name="Existing User",
            hashed_password="$2b$12$fakehash",
            disabled=False,
            role=UserRole.USER,
            preferences={},
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_returns_existing_user(self, decoded_token, existing_user):
        """Test that an existing user is returned when found by email."""
        mock_db = MagicMock()
        mock_db.get_user_by_email = AsyncMock(return_value=existing_user)
        mock_db.update_user = AsyncMock(return_value=True)

        with patch("firebase_auth.get_db_manager", return_value=mock_db):
            from firebase_auth import get_or_create_firebase_user
            result = await get_or_create_firebase_user(decoded_token)

            assert result.id == "existing-user-id"
            assert result.email == "newuser@example.com"
            mock_db.get_user_by_email.assert_called_once_with("newuser@example.com")

    @pytest.mark.asyncio
    async def test_creates_new_user(self, decoded_token):
        """Test that a new user is created when not found."""
        created_user = UserInDB(
            id="new-user-id",
            username="newuser",
            email="newuser@example.com",
            full_name="New User",
            hashed_password="$2b$12$fakehash",
            disabled=False,
            role=UserRole.USER,
            preferences={},
            metadata={"firebase_uid": "firebase-uid-123", "auth_provider": "google"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_db = MagicMock()
        mock_db.get_user_by_email = AsyncMock(return_value=None)
        mock_db.get_user_by_username = AsyncMock(return_value=None)
        mock_db.create_user = AsyncMock(return_value="new-user-id")
        mock_db.get_user = AsyncMock(return_value=created_user)

        with patch("firebase_auth.get_db_manager", return_value=mock_db):
            from firebase_auth import get_or_create_firebase_user
            result = await get_or_create_firebase_user(decoded_token)

            assert result.email == "newuser@example.com"
            assert result.full_name == "New User"
            mock_db.create_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_email_raises(self):
        """Test that a token without email raises an error."""
        token_no_email = {"uid": "uid-123", "name": "No Email"}

        mock_db = MagicMock()

        with patch("firebase_auth.get_db_manager", return_value=mock_db):
            from firebase_auth import get_or_create_firebase_user
            with pytest.raises(AuthenticationError, match="email"):
                await get_or_create_firebase_user(token_no_email)
