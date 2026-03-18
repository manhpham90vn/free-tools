"""Base adapter class for multi-provider support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schema import InternalRequest, InternalStreamEvent


class BaseAdapter(ABC):
    """
    Abstract base class for provider adapters.

    Each provider (Gemini, OpenAI, Claude, Ollama, etc.) implements this class
    to handle conversion between its native format and the internal schema.

    Usage:
        class OllamaAdapter(BaseAdapter):
            name = "ollama"

            def parse_request(self, body: bytes, model: str) -> InternalRequest:
                ...

            def format_request(self, req: InternalRequest) -> dict:
                ...

            def parse_stream_event(self, event: dict, state: Any) -> InternalStreamEvent | None:
                ...

            def format_stream_event(self, event: InternalStreamEvent, state: Any) -> str | None:
                ...

            def create_stream_state(self) -> Any:
                return OllamaStreamState()

    Then register with: register("ollama", OllamaAdapter)
    """

    name: str = ""

    @abstractmethod
    def parse_request(self, body: bytes, model: str) -> InternalRequest:
        """
        Parse a request from this provider's format into InternalRequest.

        Args:
            body: Raw request body (bytes)
            model: Model name extracted from URL/path

        Returns:
            InternalRequest in unified format
        """
        ...

    @abstractmethod
    def format_request(self, req: InternalRequest) -> dict:
        """
        Format an InternalRequest into this provider's native request format.

        Args:
            req: InternalRequest in unified format

        Returns:
            Dict in provider's native format (will be JSON-encoded)
        """
        ...

    @abstractmethod
    def create_stream_state(self) -> Any:
        """
        Create a provider-specific stream state object.

        This object tracks state across streaming events (e.g., accumulated
        tool call arguments).

        Returns:
            Any provider-specific state object
        """
        ...

    @abstractmethod
    def parse_stream_event(self, event: dict, state: Any) -> InternalStreamEvent | None:
        """
        Parse a streaming event from this provider into InternalStreamEvent.

        Args:
            event: Raw event dict from provider's SSE stream
            state: Provider-specific stream state

        Returns:
            InternalStreamEvent or None to skip this event
        """
        ...

    @abstractmethod
    def format_stream_event(self, event: InternalStreamEvent, state: Any) -> str | None:
        """
        Format an InternalStreamEvent into this provider's SSE format.

        Args:
            event: InternalStreamEvent in unified format
            state: Provider-specific stream state

        Returns:
            SSE data line (without "data: " prefix) or None to skip
        """
        ...

    def get_headers(self, api_key: str) -> dict[str, str]:
        """
        Get provider-specific HTTP headers.

        Args:
            api_key: API key for authentication

        Returns:
            Dict of HTTP headers
        """
        return {"Content-Type": "application/json"}

    def get_endpoint(self, base_url: str) -> str:
        """
        Get the full API endpoint path.

        Args:
            base_url: Base URL from config

        Returns:
            Full endpoint path (e.g., "/v1/chat/completions")
        """
        return ""


import uuid  # noqa: E402 - used by StreamState default


class StreamState:
    """Base stream state with common fields."""

    def __init__(self) -> None:
        self.model: str = ""
        self.message_id: str = ""
        self.response_id: str = f"resp_{uuid.uuid4().hex[:12]}"
