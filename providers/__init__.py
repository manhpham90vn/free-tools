"""Multi-provider adapter registry.

Usage:
    from providers import get_adapter

    # Get adapter by name
    gemini = get_adapter("gemini")
    openai = get_adapter("openai")
    claude = get_adapter("claude")

    # Parse request from source
    internal = gemini.parse_request(body, model)

    # Format for target
    target_req = openai.format_request(internal)

    # Add custom adapter:
    from providers import register
    register("ollama", OllamaAdapter)
"""

from .base import BaseAdapter
from .gemini import GeminiAdapter
from .openai import OpenAIAdapter
from .claude import ClaudeAdapter

# Registry of adapters
_ADAPTERS: dict[str, type[BaseAdapter]] = {}


def register(name: str, adapter_cls: type[BaseAdapter]) -> None:
    """
    Register a new adapter.

    Args:
        name: Provider name (e.g., "gemini", "openai", "ollama")
        adapter_cls: Adapter class that extends BaseAdapter
    """
    _ADAPTERS[name] = adapter_cls


def get_adapter(name: str) -> BaseAdapter:
    """
    Get an adapter instance by name.

    Args:
        name: Provider name

    Returns:
        BaseAdapter instance

    Raises:
        KeyError: If adapter not found
    """
    if name not in _ADAPTERS:
        raise KeyError(f"Unknown adapter: {name}. Available: {list(_ADAPTERS.keys())}")
    return _ADAPTERS[name]()


def list_adapters() -> list[str]:
    """Get list of registered adapter names."""
    return list(_ADAPTERS.keys())


# Auto-register built-in adapters
register("gemini", GeminiAdapter)
register("openai", OpenAIAdapter)
register("claude", ClaudeAdapter)


__all__ = [
    "BaseAdapter",
    "GeminiAdapter",
    "OpenAIAdapter",
    "ClaudeAdapter",
    "register",
    "get_adapter",
    "list_adapters",
]
