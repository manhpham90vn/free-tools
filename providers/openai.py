"""OpenAI adapter."""

from __future__ import annotations

import json
import uuid

from .base import BaseAdapter, StreamState
from .schema import (
    InternalMessage,
    InternalRequest,
    InternalStreamEvent,
    ToolCall,
    ToolDef,
    ToolResult,
)


class OpenAIStreamState(StreamState):
    """OpenAI-specific stream state."""

    def __init__(self) -> None:
        super().__init__()
        self.tool_call_accum: dict[int, dict] = {}  # index → {id, name, arguments}
        self.current_tool_call_index: int = 0


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI Chat Completions API format."""

    name = "openai"

    def create_stream_state(self) -> OpenAIStreamState:
        return OpenAIStreamState()

    def parse_request(self, body: bytes, model: str) -> InternalRequest:
        """Parse OpenAI request into InternalRequest."""
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
                # Handle content blocks (text, images, tool_calls)
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            internal_msg.content = block.get("text")
                        elif block_type == "tool_calls":
                            # tool_calls is an array
                            for tc in block.get("tool_calls", []):
                                internal_msg.tool_calls.append(
                                    ToolCall(
                                        id=tc.get(
                                            "id", f"call_{uuid.uuid4().hex[:12]}"
                                        ),
                                        name=tc.get("function", {}).get("name", ""),
                                        arguments=tc.get("function", {}).get(
                                            "arguments", {}
                                        ),
                                    )
                                )
                        elif block_type == "tool_result":
                            internal_msg.tool_results.append(
                                ToolResult(
                                    tool_call_id=block.get("tool_call_id", ""),
                                    content=block.get("content", ""),
                                )
                            )

            # Also check for tool_calls at message level (older format)
            for tc in msg.get("tool_calls", []):
                internal_msg.tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", {}),
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
            func = tool.get("function", {})
            tools.append(
                ToolDef(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters=func.get("parameters", {}),
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
        """Format InternalRequest into OpenAI Chat Completions format."""
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
                            "type": "tool_calls",
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                    )
                for tr in msg.tool_results:
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_call_id": tr.tool_call_id,
                            "content": tr.content,
                        }
                    )
                content = blocks

            messages.append({"role": msg.role, "content": content})

        # Build request
        openai_req: dict = {
            "model": req.model,
            "messages": messages,
            "stream": req.stream,
        }

        if req.system:
            # OpenAI uses a special "system" role message
            openai_req["messages"] = [
                {"role": "system", "content": req.system}
            ] + messages

        if req.temperature is not None:
            openai_req["temperature"] = req.temperature

        if req.max_tokens:
            openai_req["max_tokens"] = req.max_tokens

        if req.tools:
            openai_req["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in req.tools
            ]

        return openai_req

    def parse_stream_event(
        self, event: dict, state: OpenAIStreamState
    ) -> InternalStreamEvent | None:
        """Parse OpenAI SSE event into InternalStreamEvent."""
        # OpenAI SSE format: data: {"choices":[{"delta":{...},"index":0,"finish_reason":null}]}
        choices = event.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        index = choice.get("index", 0)
        finish_reason = choice.get("finish_reason")

        # Content delta
        if "content" in delta and delta["content"]:
            return InternalStreamEvent(type="text", text=delta["content"])

        # Tool call delta
        if "tool_calls" in delta:
            tool_calls = delta["tool_calls"]
            if tool_calls:
                tc = tool_calls[0]
                func = tc.get("function", {})

                # Initialize accumulator if needed
                if index not in state.tool_call_accum:
                    state.tool_call_accum[index] = {
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                        "name": func.get("name", ""),
                        "arguments": "",
                    }

                # Accumulate arguments
                if "arguments" in func:
                    state.tool_call_accum[index]["arguments"] += func["arguments"]

                # If this is a new tool call with name, emit it
                if func.get("name"):
                    return InternalStreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                            name=func.get("name", ""),
                            arguments={},
                        ),
                    )

        # Finish reason
        if finish_reason:
            reason_map = {
                "stop": "stop",
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "content_filter": "stop",
            }
            return InternalStreamEvent(
                type="done",
                finish_reason=reason_map.get(finish_reason, "stop"),
            )

        return None

    def format_stream_event(
        self, event: InternalStreamEvent, state: OpenAIStreamState
    ) -> str | None:
        """Format InternalStreamEvent into OpenAI SSE format."""
        if event.type == "text" and event.text:
            return json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": event.text},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ]
                }
            )

        if event.type == "thinking" and event.thinking:
            # OpenAI doesn't have native thinking - convert to text
            return json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": event.thinking},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ]
                }
            )

        if event.type == "tool_call" and event.tool_call:
            tc = event.tool_call
            return json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "function": {
                                            "name": tc.name,
                                            "arguments": json.dumps(tc.arguments),
                                        },
                                    }
                                ]
                            },
                            "index": 0,
                            "finish_reason": None,
                        }
                    ]
                }
            )

        if event.type == "done":
            reason_map = {
                "stop": "stop",
                "max_tokens": "length",
                "tool_use": "tool_calls",
            }
            return json.dumps(
                {
                    "choices": [
                        {
                            "delta": {},
                            "index": 0,
                            "finish_reason": reason_map.get(
                                event.finish_reason or "stop", "stop"
                            ),
                        }
                    ]
                }
            )

        return None

    def get_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def get_endpoint(self, base_url: str) -> str:
        return "/v1/chat/completions"
