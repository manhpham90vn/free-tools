"""
Multi-Provider Adapter Registry.

This module provides the adapter registry system that allows the MITM proxy
to translate between different LLM provider formats.

The registry pattern allows easy addition of new providers without modifying
the core proxy code.

Usage:
    from providers import get_adapter

    # Get adapter by name
    gemini = get_adapter("gemini")
    openai = get_adapter("openai")
    claude = get_adapter("claude")

    # Parse request from source provider
    internal = gemini.parse_request(body, model)

    # Format for target provider
    target_req = openai.format_request(internal)

    # Register custom adapter
    from providers import register
    register("ollama", OllamaAdapter)

How it works:
- Each provider has an Adapter class that handles format conversion
- The adapter converts the provider's native format to/from our Internal schema
- The Internal schema is a unified format that all adapters use as common ground
- This allows any source → any target conversion

Example flow:
    Client (Gemini format)
        ↓
    GeminiAdapter.parse_request() → InternalRequest
        ↓
    OpenAIAdapter.format_request() → OpenAI format
        ↓
    OpenAI API
        ↓
    OpenAIAdapter.parse_stream_event() → InternalStreamEvent
        ↓
    GeminiAdapter.format_stream_event() → Gemini SSE
        ↓
    Client receives Gemini-format response
"""

# === Internal imports ===
from .base import BaseAdapter  # Base class for all adapters
from .gemini import GeminiAdapter  # Google Gemini/Antigravity adapter
from .openai import OpenAIAdapter  # OpenAI API adapter
from .claude import ClaudeAdapter  # Anthropic Claude API adapter


# =============================================================================
# REGISTRY
# =============================================================================

# Internal registry mapping provider names to their adapter classes
# This is the central lookup table for all registered adapters
_ADAPTERS: dict[str, type[BaseAdapter]] = {}


def register(name: str, adapter_cls: type[BaseAdapter]) -> None:
    """
    Register a new adapter with the registry.

    This allows adding new providers without modifying core code.
    After registration, the adapter can be retrieved using get_adapter().

    Args:
        name: Provider identifier (e.g., "gemini", "openai", "ollama")
        adapter_cls: Adapter class that extends BaseAdapter

    Example:
        register("ollama", OllamaAdapter)
        adapter = get_adapter("ollama")
    """
    _ADAPTERS[name] = adapter_cls


def get_adapter(name: str) -> BaseAdapter:
    """
    Get an adapter instance by provider name.

    Retrieves the registered adapter for the given provider and
    returns a new instance of it.

    Args:
        name: Provider identifier (e.g., "gemini", "openai", "claude")

    Returns:
        An instance of the requested adapter

    Raises:
        KeyError: If the provider is not registered

    Example:
        adapter = get_adapter("claude")
        internal_req = adapter.parse_request(body, model)
    """
    if name not in _ADAPTERS:
        raise KeyError(f"Unknown adapter: {name}. Available: {list(_ADAPTERS.keys())}")
    # Create a new instance of the adapter
    return _ADAPTERS[name]()


def list_adapters() -> list[str]:
    """
    Get a list of all registered adapter names.

    Returns:
        List of provider identifiers that can be used with get_adapter()
    """
    return list(_ADAPTERS.keys())


# =============================================================================
# AUTO-REGISTRATION OF BUILT-IN ADAPTERS
# =============================================================================

# Register the built-in adapters automatically when this module is imported
# These are the supported providers out of the box
register("gemini", GeminiAdapter)  # Google Gemini / Antigravity
register("openai", OpenAIAdapter)  # OpenAI Chat Completions API
register("claude", ClaudeAdapter)  # Anthropic Claude Messages API


# =============================================================================
# EXPORTS
# =============================================================================

# Public API - these are the symbols exported when using `from providers import ...`
__all__ = [
    "BaseAdapter",  # Abstract base class for adapters
    "GeminiAdapter",  # Google Gemini adapter
    "OpenAIAdapter",  # OpenAI adapter
    "ClaudeAdapter",  # Anthropic Claude adapter
    "register",  # Function to register new adapters
    "get_adapter",  # Function to get adapter instances
    "list_adapters",  # Function to list available adapters
]
