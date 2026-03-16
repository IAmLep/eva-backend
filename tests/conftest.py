"""
Test fixtures for EVA backend tests.

Sets up the FastAPI test client and mock dependencies.
"""

import os
import sys

# Ensure SECRET_KEY is set before importing any app modules
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("APP_ENV", "development")

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager for tests."""
    from database import DatabaseManager

    mock = MagicMock(spec=DatabaseManager)
    mock.db = None  # Use in-memory mode
    mock.in_memory_db = {
        "users": {}, "memories": {}, "conversations": {},
        "sync_states": {}, "api_keys": {}, "categories": {}, "secrets": {}
    }

    # Set up commonly used async methods
    mock.create_user = AsyncMock(return_value="test-user-id")
    mock.get_user = AsyncMock(return_value=None)
    mock.get_user_by_username = AsyncMock(return_value=None)
    mock.get_user_by_email = AsyncMock(return_value=None)
    mock.update_user = AsyncMock(return_value=True)
    mock.create_conversation = AsyncMock(return_value="test-conv-id")
    mock.get_conversation = AsyncMock(return_value=None)
    mock.get_user_conversations = AsyncMock(return_value=[])
    mock.add_message_to_conversation = AsyncMock(return_value=True)
    mock.get_conversation_messages = AsyncMock(return_value=[])

    return mock


@pytest.fixture
def test_user():
    """Create a test user object."""
    from models import User, UserRole
    from datetime import datetime, timezone

    return User(
        id="test-user-123",
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        disabled=False,
        role=UserRole.USER,
        preferences={},
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def test_user_in_db():
    """Create a test UserInDB object."""
    from models import UserInDB, UserRole
    from datetime import datetime, timezone

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
