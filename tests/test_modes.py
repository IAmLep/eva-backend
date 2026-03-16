"""
Tests for the mode switching system.
"""

import pytest
from modes import (
    AssistantMode,
    ModeConfig,
    get_mode_config,
    get_available_modes,
    get_all_modes,
    get_system_prompt,
    MODE_REGISTRY,
    GameState,
)


class TestAssistantMode:
    """Tests for the AssistantMode enum."""

    def test_chat_mode_value(self):
        assert AssistantMode.CHAT.value == "chat"

    def test_game_mode_value(self):
        assert AssistantMode.GAME.value == "game"

    def test_mode_from_string(self):
        assert AssistantMode("chat") == AssistantMode.CHAT
        assert AssistantMode("game") == AssistantMode.GAME

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            AssistantMode("invalid")


class TestModeConfig:
    """Tests for mode configuration."""

    def test_chat_mode_config(self):
        config = get_mode_config(AssistantMode.CHAT)
        assert config.mode == AssistantMode.CHAT
        assert config.display_name == "Chat"
        assert config.is_available is True
        assert "memory" in config.features

    def test_game_mode_config(self):
        config = get_mode_config(AssistantMode.GAME)
        assert config.mode == AssistantMode.GAME
        assert config.display_name == "Game"
        assert config.is_available is False  # Not yet implemented
        assert "narration" in config.features

    def test_invalid_mode_config(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_mode_config("nonexistent")


class TestModeRegistry:
    """Tests for mode registry functions."""

    def test_get_available_modes_returns_only_available(self):
        available = get_available_modes()
        assert len(available) >= 1
        for mode in available:
            assert mode.is_available is True

    def test_chat_is_available(self):
        available = get_available_modes()
        mode_values = [m.mode for m in available]
        assert AssistantMode.CHAT in mode_values

    def test_get_all_modes_includes_unavailable(self):
        all_modes = get_all_modes()
        assert len(all_modes) >= 2  # Chat + Game at minimum

    def test_mode_registry_has_expected_entries(self):
        assert AssistantMode.CHAT in MODE_REGISTRY
        assert AssistantMode.GAME in MODE_REGISTRY


class TestSystemPrompts:
    """Tests for system prompts."""

    def test_chat_system_prompt_contains_eva(self):
        prompt = get_system_prompt(AssistantMode.CHAT)
        assert "EVA" in prompt

    def test_chat_system_prompt_mentions_emotional_awareness(self):
        prompt = get_system_prompt(AssistantMode.CHAT)
        assert "emotional" in prompt.lower()

    def test_game_system_prompt_contains_game_master(self):
        prompt = get_system_prompt(AssistantMode.GAME)
        assert "Game Master" in prompt

    def test_system_prompts_are_non_empty(self):
        for mode in AssistantMode:
            prompt = get_system_prompt(mode)
            assert len(prompt) > 50


class TestGameState:
    """Tests for the GameState model."""

    def test_game_state_creation(self):
        state = GameState(game_id="game-1", user_id="user-1")
        assert state.game_id == "game-1"
        assert state.user_id == "user-1"
        assert state.is_active is True
        assert state.characters == []
        assert state.clues == []
        assert state.inventory == []

    def test_game_state_with_data(self):
        state = GameState(
            game_id="game-1",
            user_id="user-1",
            scenario="Murder Mystery",
            characters=[{"name": "Detective", "role": "protagonist"}],
            current_scene="The Library",
        )
        assert state.scenario == "Murder Mystery"
        assert len(state.characters) == 1
        assert state.current_scene == "The Library"
