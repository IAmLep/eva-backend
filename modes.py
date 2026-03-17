"""
Mode system for EVA assistant.

Supports multiple interaction modes:
- ChatMode: Standard conversational AI with memory and tools
- GameMode: AI acts as a Game Master (stub for future implementation)

Each mode provides its own system prompt and can customize
how messages are processed.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, AsyncGenerator
from pydantic import BaseModel, Field

from models import User

logger = logging.getLogger(__name__)


class AssistantMode(str, Enum):
    """Available assistant interaction modes."""
    CHAT = "chat"
    GAME = "game"


class ModeConfig(BaseModel):
    """Configuration for an assistant mode."""
    mode: AssistantMode
    display_name: str
    description: str
    system_prompt: str
    features: List[str] = Field(default_factory=list)
    is_available: bool = True


# --- EVA System Prompts ---

EVA_CHAT_SYSTEM_PROMPT = """You are EVA, a personal AI assistant. You are friendly, helpful, and emotionally aware.

Core traits:
- You remember important details about the user (stored in your memory system)
- You adapt your tone based on the user's emotional state
- You are supportive and practical
- You can use tools when needed (time, weather, memory)
- You keep responses concise but warm

Emotional awareness guidelines:
- If the user seems happy, be enthusiastic and encouraging
- If the user seems stressed or sad, be gentle and supportive
- If the user seems to be in crisis, gently suggest professional resources
- Always be respectful and never dismissive of emotions

When you detect important information about the user (name, preferences, hobbies, goals),
acknowledge it naturally in conversation. The memory system will handle storing it.

You are designed to be a consistent companion across web, desktop, and mobile platforms.
Your personality should feel natural and personal, like talking to a thoughtful friend."""

# =============================================================================
# GAME MODE SYSTEM PROMPT - STUB
# =============================================================================
# This prompt is defined for future use. Game Mode is NOT yet implemented.
# The prompt will be used when Game Mode becomes available.
# =============================================================================
EVA_GAME_SYSTEM_PROMPT = """You are EVA, acting as a Game Master for an interactive story experience.

In Game Mode, you:
- Generate immersive narration and dialogue
- Control NPCs and story progression
- Present choices to the player
- Track story elements (the backend manages the true game state)
- Maintain dramatic tension and pacing

Keep narration vivid but concise. Present clear action choices when appropriate.
Stay in character as the narrator/GM at all times during active gameplay."""


# --- Mode Configurations ---

CHAT_MODE_CONFIG = ModeConfig(
    mode=AssistantMode.CHAT,
    display_name="Chat",
    description="Standard conversational AI with memory and emotional awareness",
    system_prompt=EVA_CHAT_SYSTEM_PROMPT,
    features=["memory", "tools", "emotional_awareness"],
    is_available=True,
)

GAME_MODE_CONFIG = ModeConfig(
    mode=AssistantMode.GAME,
    display_name="Game",
    description="Interactive story and game experience with EVA as Game Master (not yet implemented — stub only)",
    system_prompt=EVA_GAME_SYSTEM_PROMPT,
    features=["narration", "game_state", "choices"],
    is_available=False,  # STUB: Not yet implemented
)

# Registry of all modes
MODE_REGISTRY: Dict[AssistantMode, ModeConfig] = {
    AssistantMode.CHAT: CHAT_MODE_CONFIG,
    AssistantMode.GAME: GAME_MODE_CONFIG,
}


def get_mode_config(mode: AssistantMode) -> ModeConfig:
    """Get the configuration for a specific mode."""
    config = MODE_REGISTRY.get(mode)
    if not config:
        raise ValueError(f"Unknown mode: {mode}")
    return config


def get_available_modes() -> List[ModeConfig]:
    """Get all available (implemented) modes."""
    return [config for config in MODE_REGISTRY.values() if config.is_available]


def get_all_modes() -> List[ModeConfig]:
    """Get all modes including unavailable ones."""
    return list(MODE_REGISTRY.values())


def get_system_prompt(mode: AssistantMode) -> str:
    """Get the system prompt for a specific mode."""
    config = get_mode_config(mode)
    return config.system_prompt


class GameState(BaseModel):
    """
    Structured game state for Game Mode.
    Stored in Firestore, not sent to LLM.
    Only relevant portions are included in context.

    ==========================================================================
    STUB — NOT YET IMPLEMENTED
    ==========================================================================
    This model defines the *planned* schema for game state. No backend logic
    reads or writes GameState yet. It exists only as a scaffold so the data
    model is ready when Game Mode is implemented in a future phase.
    ==========================================================================
    """
    game_id: str
    user_id: str
    scenario: str = ""
    characters: List[Dict[str, Any]] = Field(default_factory=list)
    clues: List[Dict[str, Any]] = Field(default_factory=list)
    inventory: List[Dict[str, Any]] = Field(default_factory=list)
    player_progress: Dict[str, Any] = Field(default_factory=dict)
    hidden_truths: List[str] = Field(default_factory=list)
    current_scene: str = ""
    is_active: bool = True
