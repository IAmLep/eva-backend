"""
Tests for the API endpoints (conversation and modes).
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from models import User, UserInDB, UserRole
from datetime import datetime, timezone


@pytest.fixture
def test_user():
    """Create a test user for authentication."""
    return UserInDB(
        id="test-user-123",
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        hashed_password="$2b$12$fakehashfortest",
        disabled=False,
        role=UserRole.USER,
        preferences={},
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def app(test_user):
    """Create a test FastAPI app with mocked dependencies."""
    with patch("database.FIRESTORE_AVAILABLE", False):
        from main import app
        from auth import get_current_user, get_current_active_user
        from rate_limiter import rate_limiter_dependency

        async def override_get_current_user():
            return test_user

        async def override_get_current_active_user():
            return test_user

        async def override_rate_limiter():
            pass  # No-op for tests

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_current_active_user] = override_get_current_active_user
        app.dependency_overrides[rate_limiter_dependency] = override_rate_limiter
        yield app
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_endpoint(app):
    """Test the health check endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_get_modes(app):
    """Test the modes endpoint returns available modes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/conversation/modes",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        modes = response.json()
        assert isinstance(modes, list)
        assert len(modes) >= 2

        # Find chat mode
        chat_modes = [m for m in modes if m["mode"] == "chat"]
        assert len(chat_modes) == 1
        assert chat_modes[0]["is_available"] is True
        assert chat_modes[0]["display_name"] == "Chat"

        # Find game mode
        game_modes = [m for m in modes if m["mode"] == "game"]
        assert len(game_modes) == 1
        assert game_modes[0]["is_available"] is False


@pytest.mark.asyncio
async def test_health_endpoint(app):
    """Test the health check endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_get_modes(app):
    """Test the modes endpoint returns available modes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/conversation/modes",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        modes = response.json()
        assert isinstance(modes, list)
        assert len(modes) >= 2

        # Find chat mode
        chat_modes = [m for m in modes if m["mode"] == "chat"]
        assert len(chat_modes) == 1
        assert chat_modes[0]["is_available"] is True
        assert chat_modes[0]["display_name"] == "Chat"

        # Find game mode
        game_modes = [m for m in modes if m["mode"] == "game"]
        assert len(game_modes) == 1
        assert game_modes[0]["is_available"] is False
