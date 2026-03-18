"""
Claude (Anthropic) Adapter.

This adapter handles conversion between Anthropic's Claude Messages API format
and the internal unified schema. The Claude API is a common target provider
for this MITM proxy since it offers powerful models (Claude Opus, Sonnet, Haiku).

Claude format uses:
- messages[] with content blocks (text, tool_use, tool_result)
- system as a top-level string (not a message role)
- tools[] with input_schema for function calling
- SSE streaming with event types: message_start, content_block_start/delta/stop, message_delta

Key differences from OpenAI:
- Tool calls use "tool_use" type with "input" (not "function" with "arguments")
- Tool results use "tool_result" with "tool_use_id" (not "tool_call_id")
- System prompt is a top-level field (not a "system" role message)
- Uses x-api-key header (not Bearer token)
"""

from __future__ import annotations

import json
import time

from .base import BaseAdapter, StreamState
from .schema import (
    InternalMessage,
    InternalRequest,
    InternalStreamEvent,
    ToolCall,
    ToolDef,
    ToolResult,
    Usage,
)


class ClaudeStreamState(StreamState):
    """
    Claude-specific stream state for tracking streaming events.

    Extends StreamState with Claude-specific fields:
    - tool_call_accum: Accumulates tool call arguments across streaming chunks
    - in_thinking: Whether we're currently in a thinking block (extended thinking)
    - usage: Token usage including cache tokens
    """

    def __init__(self) -> None:
        super().__init__()
        # Maps index -> {id, name, arguments} for accumulating tool call data
        self.tool_call_accum: dict[int, dict] = {}
        # Tracks whether we're inside a "thinking" content block
        self.in_thinking: bool = False
        # Usage stats (includes cache tokens unique to Claude)
        self.usage: Usage | None = None


class ClaudeAdapter(BaseAdapter):
    """
    Adapter for Anthropic's Claude Messages API format.

    Handles conversion between:
    - Claude format: messages[] with content blocks, system string, tools[] with input_schema
    - Internal format: messages[], system, temperature, tools
    """

    name = "claude"

    def create_stream_state(self) -> ClaudeStreamState:
        """Create a new ClaudeStreamState for tracking streaming."""
        return ClaudeStreamState()

    def parse_request(self, body: bytes, model: str) -> InternalRequest:
        """
        Parse Claude request into InternalRequest (for when Claude is source).

        Claude's Messages API format:
        - {"model": "...", "messages": [...], "system": "...", "tools": [...]}
        - Content can be string or array of blocks (text, tool_use, tool_result)

        Args:
            body: Raw request body bytes (JSON)
            model: Model name from URL (may be overridden by body)

        Returns:
            InternalRequest in unified format
        """
        raw = json.loads(body)

        # Extract top-level fields
        model = raw.get("model", model)
        system = raw.get("system")  # Claude uses top-level system, not a message
        temperature = raw.get("temperature")
        max_tokens = raw.get("max_tokens", 16384)

        # === Parse messages ===
        messages: list[InternalMessage] = []
        for msg in raw.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content")

            internal_msg = InternalMessage(role=role)

            # === Handle content ===
            # Claude content can be either a string or an array of content blocks
            if isinstance(content, str):
                # Simple string content
                internal_msg.content = content
            elif isinstance(content, list):
                # Content blocks: text, tool_use, tool_result
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        # === Text block ===
                        if block_type == "text":
                            internal_msg.content = block.get("text")
                        # === Tool use block (tool call from assistant) ===
                        # Claude uses "tool_use" with "input" field
                        elif block_type == "tool_use":
                            internal_msg.tool_calls.append(
                                ToolCall(
                                    id=block.get("id", ""),
                                    name=block.get("name", ""),
                                    arguments=block.get("input", {}),
                                )
                            )
                        # === Tool result block (result from user) ===
                        # Claude uses "tool_result" with "tool_use_id"
                        elif block_type == "tool_result":
                            internal_msg.tool_results.append(
                                ToolResult(
                                    tool_call_id=block.get("tool_use_id", ""),
                                    content=block.get("content", ""),
                                )
                            )

            # Only add non-empty messages
            if (
                internal_msg.content
                or internal_msg.tool_calls
                or internal_msg.tool_results
            ):
                messages.append(internal_msg)

        # === Parse tools ===
        # Claude tools format: [{"name": "...", "description": "...", "input_schema": {...}}]
        tools: list[ToolDef] = []
        for tool in raw.get("tools", []):
            tools.append(
                ToolDef(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    # Claude uses "input_schema" instead of "parameters"
                    parameters=tool.get("input_schema", {}),
                )
            )

        return InternalRequest(
            model=model,
            messages=messages,
            system=system,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=raw.get("stream", True),
        )

    def format_request(self, req: InternalRequest) -> dict:
        """
        Format InternalRequest into Claude Messages API format.

        This is typically used when Claude is the TARGET provider.

        Converts:
        - InternalMessage[] -> messages[] (with content blocks if tools present)
        - InternalRequest.system -> top-level "system" field
        - InternalRequest.tools -> tools[] with input_schema

        Args:
            req: InternalRequest in unified format

        Returns:
            Dictionary in Claude API format
        """
        messages: list[dict] = []

        for msg in req.messages:
            content: str | list = msg.content or ""

            # === Build content blocks if there are tool calls/results ===
            # Claude requires content blocks format when tools are involved
            if msg.tool_calls or msg.tool_results:
                blocks: list[dict] = []
                # Add text content as first block if present
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                # Add tool calls as "tool_use" blocks
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,  # Claude uses "input" not "arguments"
                        }
                    )
                # Add tool results as "tool_result" blocks
                for tr in msg.tool_results:
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,  # Claude uses "tool_use_id"
                            "content": tr.content,
                        }
                    )
                content = blocks

            messages.append({"role": msg.role, "content": content})

        # === Build Claude request ===
        claude_req: dict = {
            "model": req.model,
            "messages": messages,
            "stream": req.stream,
        }

        # === System prompt ===
        # Claude uses a top-level "system" field (not a system role message)
        if req.system:
            claude_req["system"] = req.system

        # === Optional parameters ===
        if req.temperature is not None:
            claude_req["temperature"] = req.temperature

        # max_tokens is required for Claude API
        claude_req["max_tokens"] = req.max_tokens

        # === Tools ===
        # Claude uses "input_schema" instead of "parameters"
        if req.tools:
            claude_req["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in req.tools
            ]

        return claude_req

    def parse_stream_event(
        self, event: dict, state: ClaudeStreamState
    ) -> InternalStreamEvent | None:
        """
        Parse Claude SSE event into InternalStreamEvent.

        This is used when Claude is the TARGET provider.

        Claude SSE event types:
        - message_start: Initial message metadata (id, model)
        - content_block_start: Start of a content block (text, thinking, tool_use)
        - content_block_delta: Incremental content (text_delta, thinking_delta, input_json_delta)
        - content_block_stop: End of a content block
        - message_delta: Final usage data and stop reason
        - message_stop: Stream complete

        Args:
            event: Raw event dict from Claude SSE
            state: Stream state for accumulating tool call arguments

        Returns:
            InternalStreamEvent or None to skip
        """
        event_type = event.get("type", "")

        # === message_start: Initialize message metadata ===
        if event_type == "message_start":
            msg = event.get("message", {})
            state.message_id = msg.get("id", f"msg_{int(time.time())}")
            state.model = msg.get("model", state.model)
            return None  # No output to client yet

        # === content_block_start: Initialize a new content block ===
        elif event_type == "content_block_start":
            block = event.get("content_block", {})
            block_type = block.get("type", "")
            index = event.get("index", 0)

            # Extended thinking block
            if block_type == "thinking":
                state.in_thinking = True
                return None

            # Tool use block - initialize accumulator for arguments
            elif block_type == "tool_use":
                state.tool_call_accum[index] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": "",  # Will be accumulated from input_json_delta events
                }
                return None

            # Text block - nothing to do yet, wait for deltas
            elif block_type == "text":
                return None

        # === content_block_delta: Incremental content updates ===
        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")
            index = event.get("index", 0)

            # Text delta: Return as text event
            if delta_type == "text_delta" and delta.get("text"):
                return InternalStreamEvent(type="text", text=delta["text"])

            # Thinking delta: Return as thinking event (Claude's extended thinking)
            elif delta_type == "thinking_delta" and delta.get("thinking"):
                return InternalStreamEvent(type="thinking", thinking=delta["thinking"])

            # Input JSON delta: Accumulate tool call arguments
            # Tool arguments come in pieces via partial_json
            elif delta_type == "input_json_delta" and delta.get("partial_json"):
                if index in state.tool_call_accum:
                    state.tool_call_accum[index]["arguments"] += delta["partial_json"]
                return None

        # === content_block_stop: End of content block ===
        elif event_type == "content_block_stop":
            state.in_thinking = False
            return None

        # === message_delta: Final usage and stop reason ===
        elif event_type == "message_delta":
            # Extract token usage (includes Claude-specific cache tokens)
            usage = event.get("usage", {})
            if usage:
                input_tokens = usage.get("input_tokens") or 0
                output_tokens = usage.get("output_tokens") or 0
                # Claude caching: cache_read = tokens read from cache (cheaper)
                cache_read = usage.get("cache_read_input_tokens") or 0
                # cache_creation = tokens written to cache (costs extra first time)
                cache_creation = usage.get("cache_creation_input_tokens") or 0
                state.usage = Usage(
                    input_tokens=input_tokens + cache_read + cache_creation,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens
                    + output_tokens
                    + cache_read
                    + cache_creation,
                )

            stop_reason = event.get("delta", {}).get("stop_reason")

            # Emit accumulated tool calls before the done event
            # Tool call arguments have been accumulating across delta events
            if state.tool_call_accum:
                for idx in sorted(state.tool_call_accum.keys()):
                    accum = state.tool_call_accum[idx]
                    args = {}
                    try:
                        # Parse accumulated JSON string into dict
                        args = json.loads(accum["arguments"])
                    except (json.JSONDecodeError, ValueError):
                        pass  # Invalid JSON, use empty dict

                    return InternalStreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=accum["id"],
                            name=accum["name"],
                            arguments=args,
                        ),
                    )

            # Emit done event with stop reason
            if stop_reason:
                # Map Claude stop reasons to internal format
                reason_map = {
                    "end_turn": "stop",  # Normal completion
                    "max_tokens": "max_tokens",  # Hit token limit
                    "tool_use": "tool_use",  # Model wants to use a tool
                    "stop_sequence": "stop",  # Hit a stop sequence
                }
                return InternalStreamEvent(
                    type="done",
                    finish_reason=reason_map.get(stop_reason, "stop"),
                    usage=state.usage,
                )

        # === message_stop: Stream complete (no action needed) ===
        elif event_type == "message_stop":
            return None

        return None

    def format_stream_event(
        self, event: InternalStreamEvent, state: ClaudeStreamState
    ) -> str | None:
        """
        Format InternalStreamEvent into Claude SSE format.

        This is used when Claude is the SOURCE provider.
        Not typically needed because we use the Anthropic SDK for streaming
        and handle events directly in the handler.

        Args:
            event: InternalStreamEvent
            state: Stream state

        Returns:
            None (not implemented - SDK handles this)
        """
        # When Claude is the target, we send to Claude API directly via SDK
        # The SDK handles SSE formatting internally
        return None

    def get_headers(self, api_key: str) -> dict[str, str]:
        """
        Get headers for Claude API requests.

        Claude uses x-api-key header for authentication (not Bearer token).
        Also requires anthropic-version header for API versioning.

        Args:
            api_key: Anthropic API key

        Returns:
            Headers dict with Content-Type, x-api-key, and anthropic-version
        """
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def get_endpoint(self, base_url: str) -> str:
        """
        Get the Claude messages endpoint path.

        Args:
            base_url: Base URL (ignored, returns fixed endpoint)

        Returns:
            Claude messages endpoint
        """
        return "/v1/messages"
