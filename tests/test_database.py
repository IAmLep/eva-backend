"""
Tests for the database manager conversation operations.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from models import Conversation


class TestDatabaseConversationOperations:
    """Tests for conversation CRUD operations in the database manager."""

    @pytest.fixture
    def db_manager(self):
        """Create a DatabaseManager in in-memory mode."""
        with patch.dict("os.environ", {"SECRET_KEY": "test-key"}):
            from database import DatabaseManager
            # Force in-memory mode by patching FIRESTORE_AVAILABLE
            with patch("database.FIRESTORE_AVAILABLE", False):
                manager = DatabaseManager()
                assert manager.db is None  # Confirm in-memory mode
                return manager

    @pytest.fixture
    def sample_conversation(self):
        """Create a sample conversation."""
        return Conversation(
            conversation_id="conv-test-123",
            user_id="user-test-456",
            summary="Test conversation",
            metadata={"mode": "chat"},
        )

    @pytest.mark.asyncio
    async def test_create_conversation(self, db_manager, sample_conversation):
        conv_id = await db_manager.create_conversation(sample_conversation)
        assert conv_id == "conv-test-123"

    @pytest.mark.asyncio
    async def test_get_conversation(self, db_manager, sample_conversation):
        await db_manager.create_conversation(sample_conversation)
        result = await db_manager.get_conversation("conv-test-123")
        assert result is not None
        assert result.conversation_id == "conv-test-123"
        assert result.user_id == "user-test-456"

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(self, db_manager):
        result = await db_manager.get_conversation("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_conversation(self, db_manager, sample_conversation):
        await db_manager.create_conversation(sample_conversation)
        success = await db_manager.update_conversation(
            "conv-test-123", {"summary": "Updated summary"}
        )
        assert success is True

        result = await db_manager.get_conversation("conv-test-123")
        assert result is not None

    @pytest.mark.asyncio
    async def test_update_nonexistent_conversation(self, db_manager):
        success = await db_manager.update_conversation(
            "nonexistent-id", {"summary": "Updated"}
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_get_user_conversations(self, db_manager):
        # Create multiple conversations
        for i in range(3):
            conv = Conversation(
                conversation_id=f"conv-{i}",
                user_id="user-1",
                summary=f"Conversation {i}",
            )
            await db_manager.create_conversation(conv)

        # Create a conversation for a different user
        other_conv = Conversation(
            conversation_id="conv-other",
            user_id="user-2",
            summary="Other user conversation",
        )
        await db_manager.create_conversation(other_conv)

        # Get conversations for user-1
        conversations = await db_manager.get_user_conversations("user-1")
        assert len(conversations) == 3
        for conv in conversations:
            assert conv.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_get_user_conversations_with_limit(self, db_manager):
        for i in range(5):
            conv = Conversation(
                conversation_id=f"conv-{i}",
                user_id="user-1",
                summary=f"Conversation {i}",
            )
            await db_manager.create_conversation(conv)

        conversations = await db_manager.get_user_conversations("user-1", limit=3)
        assert len(conversations) == 3

    @pytest.mark.asyncio
    async def test_add_message_to_conversation(self, db_manager, sample_conversation):
        await db_manager.create_conversation(sample_conversation)
        success = await db_manager.add_message_to_conversation(
            "conv-test-123", "user", "Hello EVA!"
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_get_conversation_messages(self, db_manager, sample_conversation):
        await db_manager.create_conversation(sample_conversation)

        # Add messages
        await db_manager.add_message_to_conversation(
            "conv-test-123", "user", "Hello EVA!"
        )
        await db_manager.add_message_to_conversation(
            "conv-test-123", "assistant", "Hello! How can I help you?"
        )

        messages = await db_manager.get_conversation_messages("conv-test-123")
        assert len(messages) == 2
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_get_empty_conversation_messages(self, db_manager):
        messages = await db_manager.get_conversation_messages("empty-conv")
        assert messages == []
