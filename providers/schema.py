"""
Internal Unified Schema for Multi-Provider Conversion.

This module defines the internal data structures that all adapters
convert to and from. These schemas provide a common "middle ground"
that allows any source provider to convert to any target provider.

The schema is designed to be:
- Provider-agnostic: Works for Gemini, OpenAI, Claude, Ollama, etc.
- Complete: Supports all common LLM features (messages, tools, streaming)
- Simple: Easy to convert between provider formats

Example conversion flow:
    Source Request (Gemini) → parse_request() → InternalRequest
    InternalRequest → format_request() → Target Request (OpenAI)

Data structures:
- ToolCall: A tool/function call from the model
- ToolResult: The result of a tool call (from the user)
- ToolDef: Definition of a tool the model can use
- Usage: Token usage statistics
- InternalMessage: A single message in the conversation
- InternalRequest: Complete request with model, messages, etc.
- InternalStreamEvent: Single streaming event (text, thinking, tool_call, done)
"""

from __future__ import annotations

import uuid  # For generating unique IDs
from dataclasses import dataclass, field  # Data classes for clean data structures
from typing import Any  # Generic type for tool arguments


# =============================================================================
# TOOL-RELATED SCHEMAS
# =============================================================================


@dataclass
class ToolCall:
    """
    Represents a tool/function call made by the assistant.

    When the model decides to call a tool, it outputs a ToolCall
    containing the tool name and arguments. The client then executes
    the tool and provides the results.

    Attributes:
        id: Unique identifier for this tool call (e.g., "call_abc123...")
        name: Name of the tool being called (e.g., "get_weather")
        arguments: Arguments to pass to the tool (as a dict)
    """

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate a unique ID if not provided."""
        if not self.id:
            self.id = f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class ToolResult:
    """
    Represents the result of a tool call, sent back by the user/client.

    After the client executes a tool call, it sends the result back
    to continue the conversation.

    Attributes:
        tool_call_id: ID of the tool call this result is for
        content: The result content (usually a string or JSON)
    """

    tool_call_id: str = ""
    content: str = ""


@dataclass
class ToolDef:
    """
    Tool/function definition - describes a tool the model can use.

    Tool definitions tell the model what tools are available and
    how to call them. Includes the name, description, and parameter schema.

    Attributes:
        name: Name of the tool
        description: What the tool does (for the model's context)
        parameters: JSON Schema for the tool's parameters
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# USAGE STATISTICS
# =============================================================================


@dataclass
class Usage:
    """
    Token usage information from an API response.

    Tracks how many tokens were used in the request and response.
    Useful for billing, rate limiting, and optimization.

    Attributes:
        input_tokens: Tokens in the prompt/request
        output_tokens: Tokens in the completion/response
        total_tokens: Sum of input + output tokens
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


# =============================================================================
# MESSAGE SCHEMA
# =============================================================================


@dataclass
class InternalMessage:
    """
    A single message in the conversation.

    Messages can be from:
    - "user": The human's input
    - "assistant": The AI's response (may include tool calls)
    - "tool": Results from tool executions

    Attributes:
        role: Who sent this message ("user", "assistant", or "tool")
        content: Text content of the message
        tool_calls: List of tool calls made in this message (assistant only)
        tool_results: List of tool results for this message (tool role)
    """

    role: str = "user"  # "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


# =============================================================================
# REQUEST SCHEMA
# =============================================================================


@dataclass
class InternalRequest:
    """
    Unified request format used internally by the proxy.

    This is the "middle ground" format that all provider adapters
    convert to and from. It contains all the information needed
    to make an LLM API request, regardless of the provider.

    Attributes:
        model: Model identifier (e.g., "gpt-4", "claude-opus-4-6")
        messages: List of conversation messages
        system: Optional system prompt
        tools: List of available tools (empty = no tools)
        temperature: Sampling temperature (None = provider default)
        max_tokens: Maximum tokens in response
        stream: Whether to stream the response
    """

    model: str = ""
    messages: list[InternalMessage] = field(default_factory=list)
    system: str | None = None
    tools: list[ToolDef] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int = 16384
    stream: bool = True


# =============================================================================
# STREAMING EVENT SCHEMA
# =============================================================================


@dataclass
class InternalStreamEvent:
    """
    A single streaming event in the unified format.

    When streaming responses, the API sends events as they occur.
    This schema normalizes those events across all providers.

    Event types:
    - "text": Incremental text content
    - "thinking": Extended thinking (for Claude models)
    - "tool_call": A tool call from the model
    - "tool_call_delta": Incremental tool call content
    - "done": Final event with usage and stop reason

    Attributes:
        type: Type of event ("text", "thinking", "tool_call", "done")
        text: Text content (for "text" events)
        tool_call: ToolCall object (for "tool_call" events)
        thinking: Thinking content (for "thinking" events)
        usage: Usage statistics (for "done" events)
        finish_reason: Why the stream ended ("stop", "tool_use", "max_tokens")
    """

    type: str = ""  # "text" | "thinking" | "tool_call" | "tool_call_delta" | "done"
    text: str | None = None
    tool_call: ToolCall | None = None
    thinking: str | None = None
    usage: Usage | None = None
    finish_reason: str | None = None  # "stop" | "tool_use" | "max_tokens"
