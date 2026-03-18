"""Internal unified schema for multi-provider conversion."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool/function call made by the assistant."""

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class ToolResult:
    """Result of a tool call, sent by the user."""

    tool_call_id: str = ""
    content: str = ""


@dataclass
class ToolDef:
    """Tool/function definition."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Token usage info."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class InternalMessage:
    """A single message in the conversation."""

    role: str = "user"  # "user" | "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class InternalRequest:
    """Unified request format across all providers."""

    model: str = ""
    messages: list[InternalMessage] = field(default_factory=list)
    system: str | None = None
    tools: list[ToolDef] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int = 16384
    stream: bool = True


@dataclass
class InternalStreamEvent:
    """A single streaming event in unified format."""

    type: str = ""  # "text" | "thinking" | "tool_call" | "tool_call_delta" | "done"
    text: str | None = None
    tool_call: ToolCall | None = None
    thinking: str | None = None
    usage: Usage | None = None
    finish_reason: str | None = None  # "stop" | "tool_use" | "max_tokens"
