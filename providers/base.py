"""
Base Adapter Class for Multi-Provider Support.

This module defines the abstract base class that all provider adapters
must implement. The adapter pattern allows the MITM proxy to handle
multiple LLM providers (Gemini, OpenAI, Claude, Ollama, etc.) uniformly.

Each provider adapter is responsible for:
- parse_request(): Converting the provider's native request format → InternalRequest
- format_request(): Converting InternalRequest → provider's native request format
- parse_stream_event(): Converting the provider's streaming event → InternalStreamEvent
- format_stream_event(): Converting InternalStreamEvent → provider's streaming format
- create_stream_state(): Creating a provider-specific state object for streaming
- get_headers(): Providing provider-specific HTTP headers
- get_endpoint(): Providing the API endpoint path

Example: To add a new provider (e.g., Ollama):

    class OllamaAdapter(BaseAdapter):
        name = "ollama"

        def parse_request(self, body, model):
            ...

        def format_request(self, req):
            ...

    register("ollama", OllamaAdapter)
"""

from __future__ import annotations

# === Standard library imports ===
from abc import ABC, abstractmethod  # Abstract base class support
from typing import Any  # Type hints

# === Internal imports ===
from .schema import InternalRequest, InternalStreamEvent  # Unified internal formats


class BaseAdapter(ABC):
    """
    Abstract base class for all provider adapters.

    Each LLM provider (Gemini, OpenAI, Claude, Ollama, etc.) must implement
    this class to handle conversion between its native API format
    and the internal unified schema.

    The adapter acts as a translator:
    - Source adapters: Parse incoming requests and format outgoing events
    - Target adapters: Format outgoing requests and parse incoming events

    Attributes:
        name: Provider identifier (e.g., "gemini", "openai", "claude")
    """

    name: str = ""

    @abstractmethod
    def parse_request(self, body: bytes, model: str) -> InternalRequest:
        """
        Parse a request from this provider's native format into InternalRequest.

        This is used when this provider is the SOURCE (the one sending requests).

        Args:
            body: Raw request body as bytes (will be JSON-parsed)
            model: Model name extracted from URL/path

        Returns:
            InternalRequest in the unified internal format
        """
        ...

    @abstractmethod
    def format_request(self, req: InternalRequest) -> dict:
        """
        Format an InternalRequest into this provider's native request format.

        This is used when this provider is the TARGET (the one receiving requests).

        Args:
            req: InternalRequest in unified format

        Returns:
            Dictionary in this provider's native format (will be JSON-serialized)
        """
        ...

    @abstractmethod
    def create_stream_state(self) -> Any:
        """
        Create a provider-specific stream state object.

        The state object tracks context across multiple streaming events.
        For example, it might accumulate partial tool call arguments,
        track message IDs, or count tokens.

        Returns:
            A new state object (specific to each provider's adapter)
        """
        ...

    @abstractmethod
    def parse_stream_event(self, event: dict, state: Any) -> InternalStreamEvent | None:
        """
        Parse a streaming event from this provider into InternalStreamEvent.

        This is used when this provider is the TARGET.
        The target sends streaming events that we need to convert back
        to the source format.

        Args:
            event: Raw event dictionary from the provider's SSE stream
            state: Provider-specific stream state (for accumulating context)

        Returns:
            InternalStreamEvent or None (None means skip this event)
        """
        ...

    @abstractmethod
    def format_stream_event(self, event: InternalStreamEvent, state: Any) -> str | None:
        """
        Format an InternalStreamEvent into this provider's SSE format.

        This is used when this provider is the SOURCE.
        We need to send events back in the format the client expects.

        Args:
            event: InternalStreamEvent in unified format
            state: Provider-specific stream state

        Returns:
            SSE data line string (without "data: " prefix), or None to skip
        """
        ...

    def get_headers(self, api_key: str) -> dict[str, str]:
        """
        Get provider-specific HTTP headers for API requests.

        Different providers use different authentication headers:
        - OpenAI: Authorization: Bearer <key>
        - Claude: x-api-key: <key>
        - Gemini: x-goog-api-key: <key>

        Override in subclass to add authentication headers.

        Args:
            api_key: API key for authentication

        Returns:
            Dictionary of HTTP headers (default: just Content-Type)
        """
        return {"Content-Type": "application/json"}

    def get_endpoint(self, base_url: str) -> str:
        """
        Get the API endpoint path for this provider.

        Override in subclass if the endpoint path needs to be appended
        to the base URL.

        Args:
            base_url: Base URL from configuration

        Returns:
            Endpoint path string (e.g., "/v1/chat/completions")
        """
        return ""


# === Stream State Base Class ===
import uuid  # noqa: E402 - import after class definition for readability


class StreamState:
    """
    Base stream state with common fields shared across all providers.

    Subclasses can add provider-specific fields for tracking
    additional state during streaming.

    Attributes:
        model: The model being used for the current stream
        message_id: Unique identifier for the message
        response_id: Unique identifier for the response
    """

    def __init__(self) -> None:
        self.model: str = ""
        self.message_id: str = ""
        # Generate a unique response ID for this stream
        self.response_id: str = f"resp_{uuid.uuid4().hex[:12]}"
