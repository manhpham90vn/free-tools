"""
OpenAI Adapter.

This adapter handles conversion between OpenAI's Chat Completions API format and the internal
unified schema. The OpenAI API is widely used and serves as a common format for many LLM providers.

OpenAI format uses:
- messages[] with role-based messages (system, user, assistant)
- tools[] for function calling
- Content can be string or blocks (text, tool_calls, tool_result)
- SSE streaming with choices[].delta structure
"""

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
    """
    OpenAI-specific stream state for tracking streaming events.

    Extends StreamState with OpenAI-specific fields:
    - tool_call_accum: Accumulates tool call arguments across streaming chunks
    - current_tool_call_index: Tracks which tool call we're on
    """

    def __init__(self) -> None:
        super().__init__()
        # Maps index -> {id, name, arguments} for accumulating tool call data
        self.tool_call_accum: dict[int, dict] = {}
        self.current_tool_call_index: int = 0


class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI Chat Completions API format.

    Handles conversion between:
    - OpenAI format: messages[], tools[], stream parameter
    - Internal format: messages[], system, temperature, tools
    """

    name = "openai"

    def create_stream_state(self) -> OpenAIStreamState:
        """Create a new OpenAIStreamState for tracking streaming."""
        return OpenAIStreamState()

    def parse_request(self, body: bytes, model: str) -> InternalRequest:
        """
        Parse OpenAI request into InternalRequest.

        OpenAI format:
        - {"model": "...", "messages": [...], "tools": [...], ...}
        - Messages can have string content or content blocks
        - Tools are defined in tools[] array

        Args:
            body: Raw request body bytes (JSON)
            model: Model name from URL (may be overridden by body)

        Returns:
            InternalRequest in unified format
        """
        raw = json.loads(body)

        # Extract top-level fields
        model = raw.get("model", model)
        system = raw.get("system")
        temperature = raw.get("temperature")
        max_tokens = raw.get("max_tokens", 16384)

        # === Parse messages ===
        messages: list[InternalMessage] = []
        for msg in raw.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content")

            internal_msg = InternalMessage(role=role)

            # === Handle content ===
            # OpenAI content can be either a string or an array of content blocks
            if isinstance(content, str):
                # Simple string content
                internal_msg.content = content
            elif isinstance(content, list):
                # Content blocks: text, image, tool_calls, tool_result
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        # === Text block ===
                        if block_type == "text":
                            internal_msg.content = block.get("text")
                        # === Tool calls block ===
                        elif block_type == "tool_calls":
                            # tool_calls is an array within the block
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
                        # === Tool result block ===
                        elif block_type == "tool_result":
                            internal_msg.tool_results.append(
                                ToolResult(
                                    tool_call_id=block.get("tool_call_id", ""),
                                    content=block.get("content", ""),
                                )
                            )

            # === Also check for tool_calls at message level (older format) ===
            # Some clients put tool_calls at the message level instead of in content
            for tc in msg.get("tool_calls", []):
                internal_msg.tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", {}),
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
        # OpenAI tools format: [{"type": "function", "function": {...}}]
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
        """
        Format InternalRequest into OpenAI Chat Completions format.

        This is typically used when OpenAI is the TARGET provider.

        Converts:
        - InternalMessage[] -> messages[] (with content blocks if tools present)
        - InternalRequest.system -> system role message
        - InternalRequest.tools -> tools[]

        Args:
            req: InternalRequest in unified format

        Returns:
            Dictionary in OpenAI API format
        """
        messages: list[dict] = []

        for msg in req.messages:
            content: str | list = msg.content or ""

            # === Build content blocks if there are tool calls/results ===
            # OpenAI requires content blocks format when tools are involved
            if msg.tool_calls or msg.tool_results:
                blocks: list[dict] = []
                # Add text content as first block if present
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                # Add tool calls
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
                # Add tool results
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

        # === Build OpenAI request ===
        openai_req: dict = {
            "model": req.model,
            "messages": messages,
            "stream": req.stream,
        }

        # === System message ===
        # OpenAI uses a special "system" role message at the start
        if req.system:
            openai_req["messages"] = [
                {"role": "system", "content": req.system}
            ] + messages

        # === Optional parameters ===
        if req.temperature is not None:
            openai_req["temperature"] = req.temperature

        if req.max_tokens:
            openai_req["max_tokens"] = req.max_tokens

        # === Tools ===
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
        """
        Parse OpenAI SSE event into InternalStreamEvent.

        This is used when OpenAI is the TARGET provider.
        OpenAI SSE format: {"choices": [{"delta": {...}, "index": 0, "finish_reason": null}]}

        Args:
            event: Raw event dict from OpenAI SSE
            state: Stream state for accumulating tool call arguments

        Returns:
            InternalStreamEvent or None to skip
        """
        # OpenAI SSE format
        choices = event.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        index = choice.get("index", 0)
        finish_reason = choice.get("finish_reason")

        # === Content delta ===
        # Text content arrives incrementally
        if "content" in delta and delta["content"]:
            return InternalStreamEvent(type="text", text=delta["content"])

        # === Tool call delta ===
        # Tool calls arrive in pieces: first the id+name, then arguments
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

                # Accumulate arguments (they come in chunks)
                if "arguments" in func:
                    state.tool_call_accum[index]["arguments"] += func["arguments"]

                # If this is a new tool call with name, emit it
                # The accumulated arguments will come in subsequent events
                if func.get("name"):
                    return InternalStreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                            name=func.get("name", ""),
                            arguments={},
                        ),
                    )

        # === Finish reason ===
        # Sent when the stream ends
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
        """
        Format InternalStreamEvent into OpenAI SSE format.

        This is used when OpenAI is the SOURCE provider.

        Converts:
        - InternalStreamEvent.text -> {"content": "..."}
        - InternalStreamEvent.thinking -> converted to text (OpenAI doesn't have native thinking)
        - InternalStreamEvent.tool_call -> {"tool_calls": [...]}
        - InternalStreamEvent.done -> {"finish_reason": "..."}

        Args:
            event: InternalStreamEvent in unified format
            state: Stream state

        Returns:
            SSE data line string (JSON), or None to skip
        """
        # === Text content ===
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

        # === Thinking (OpenAI doesn't have native thinking support) ===
        # Convert to text content
        if event.type == "thinking" and event.thinking:
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

        # === Tool calls ===
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

        # === Done event ===
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
        """
        Get headers for OpenAI API requests.

        OpenAI uses Bearer token authentication.

        Args:
            api_key: OpenAI API key

        Returns:
            Headers dict with Content-Type and Authorization
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def get_endpoint(self, base_url: str) -> str:
        """
        Get the OpenAI chat completions endpoint path.

        Args:
            base_url: Base URL (ignored, returns fixed endpoint)

        Returns:
            OpenAI chat completions endpoint
        """
        return "/v1/chat/completions"
