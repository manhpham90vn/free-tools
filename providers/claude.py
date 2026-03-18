"""Claude adapter."""

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
    """Claude-specific stream state."""

    def __init__(self) -> None:
        super().__init__()
        self.tool_call_accum: dict[int, dict] = {}  # index → {id, name, arguments}
        self.in_thinking: bool = False
        self.usage: Usage | None = None


class ClaudeAdapter(BaseAdapter):
    """Adapter for Claude Messages API format."""

    name = "claude"

    def create_stream_state(self) -> ClaudeStreamState:
        return ClaudeStreamState()

    def parse_request(self, body: bytes, model: str) -> InternalRequest:
        """Parse Claude request into InternalRequest (for when Claude is source)."""
        raw = json.loads(body)

        model = raw.get("model", model)
        system = raw.get("system")
        temperature = raw.get("temperature")
        max_tokens = raw.get("max_tokens", 16384)

        messages: list[InternalMessage] = []
        for msg in raw.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content")

            internal_msg = InternalMessage(role=role)

            if isinstance(content, str):
                internal_msg.content = content
            elif isinstance(content, list):
                # Handle content blocks
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            internal_msg.content = block.get("text")
                        elif block_type == "tool_use":
                            internal_msg.tool_calls.append(
                                ToolCall(
                                    id=block.get("id", ""),
                                    name=block.get("name", ""),
                                    arguments=block.get("input", {}),
                                )
                            )
                        elif block_type == "tool_result":
                            internal_msg.tool_results.append(
                                ToolResult(
                                    tool_call_id=block.get("tool_use_id", ""),
                                    content=block.get("content", ""),
                                )
                            )

            if (
                internal_msg.content
                or internal_msg.tool_calls
                or internal_msg.tool_results
            ):
                messages.append(internal_msg)

        # Tools
        tools: list[ToolDef] = []
        for tool in raw.get("tools", []):
            tools.append(
                ToolDef(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
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
        """Format InternalRequest into Claude Messages API format."""
        messages: list[dict] = []

        for msg in req.messages:
            content: str | list = msg.content or ""

            # Build content blocks if there are tool calls/results
            if msg.tool_calls or msg.tool_results:
                blocks: list[dict] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                for tr in msg.tool_results:
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": tr.content,
                        }
                    )
                content = blocks

            messages.append({"role": msg.role, "content": content})

        # Build request
        claude_req: dict = {
            "model": req.model,
            "messages": messages,
            "stream": req.stream,
        }

        if req.system:
            claude_req["system"] = req.system

        if req.temperature is not None:
            claude_req["temperature"] = req.temperature

        claude_req["max_tokens"] = req.max_tokens

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
        """Parse Claude SSE event into InternalStreamEvent."""
        event_type = event.get("type", "")

        if event_type == "message_start":
            msg = event.get("message", {})
            state.message_id = msg.get("id", f"msg_{int(time.time())}")
            state.model = msg.get("model", state.model)
            return None

        elif event_type == "content_block_start":
            block = event.get("content_block", {})
            block_type = block.get("type", "")
            index = event.get("index", 0)

            if block_type == "thinking":
                state.in_thinking = True
                return None

            elif block_type == "tool_use":
                state.tool_call_accum[index] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": "",
                }
                return None

            elif block_type == "text":
                return None

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")
            index = event.get("index", 0)

            if delta_type == "text_delta" and delta.get("text"):
                return InternalStreamEvent(type="text", text=delta["text"])

            elif delta_type == "thinking_delta" and delta.get("thinking"):
                return InternalStreamEvent(type="thinking", thinking=delta["thinking"])

            elif delta_type == "input_json_delta" and delta.get("partial_json"):
                if index in state.tool_call_accum:
                    state.tool_call_accum[index]["arguments"] += delta["partial_json"]
                return None

        elif event_type == "content_block_stop":
            state.in_thinking = False
            return None

        elif event_type == "message_delta":
            # Extract usage
            usage = event.get("usage", {})
            if usage:
                input_tokens = usage.get("input_tokens") or 0
                output_tokens = usage.get("output_tokens") or 0
                cache_read = usage.get("cache_read_input_tokens") or 0
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

            # Emit accumulated tool calls
            if state.tool_call_accum:
                for idx in sorted(state.tool_call_accum.keys()):
                    accum = state.tool_call_accum[idx]
                    args = {}
                    try:
                        args = json.loads(accum["arguments"])
                    except (json.JSONDecodeError, ValueError):
                        pass

                    return InternalStreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=accum["id"],
                            name=accum["name"],
                            arguments=args,
                        ),
                    )

            if stop_reason:
                reason_map = {
                    "end_turn": "stop",
                    "max_tokens": "max_tokens",
                    "tool_use": "tool_use",
                    "stop_sequence": "stop",
                }
                return InternalStreamEvent(
                    type="done",
                    finish_reason=reason_map.get(stop_reason, "stop"),
                    usage=state.usage,
                )

        elif event_type == "message_stop":
            return None

        return None

    def format_stream_event(
        self, event: InternalStreamEvent, state: ClaudeStreamState
    ) -> str | None:
        """Format InternalStreamEvent into Claude SSE format."""
        # This is used when Claude is the target - we send to Claude API directly
        # Not typically needed as we use the Anthropic SDK
        return None

    def get_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def get_endpoint(self, base_url: str) -> str:
        return "/v1/messages"
