"""Gemini/Antigravity adapter."""

from __future__ import annotations

import json
import uuid
from typing import Any

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


class GeminiStreamState(StreamState):
    """Gemini-specific stream state."""

    def __init__(self) -> None:
        super().__init__()
        self.tool_call_accum: dict[int, dict] = {}  # index → {id, name, arguments}
        self.in_thinking: bool = False
        self.usage: Usage | None = None


class GeminiAdapter(BaseAdapter):
    """Adapter for Gemini/Antigravity API format."""

    name = "gemini"

    def create_stream_state(self) -> GeminiStreamState:
        return GeminiStreamState()

    def parse_request(self, body: bytes, model: str) -> InternalRequest:
        """Parse Gemini/Antigravity request into InternalRequest."""
        raw = json.loads(body)

        # Unwrap Cloud Code envelope: body may be envelope or raw Gemini
        inner = raw.get("request", raw)
        contents = inner.get("contents", [])
        system_instruction = inner.get("systemInstruction", {})
        generation_config = inner.get("generationConfig", {})
        tools_raw = inner.get("tools", [])

        # --- System ---
        system_parts = []
        if system_instruction:
            for p in system_instruction.get("parts", []):
                text = p.get("text", "")
                if text:
                    system_parts.append(text)
        system = "\n".join(system_parts) if system_parts else None

        # --- Messages ---
        messages: list[InternalMessage] = []
        for content in contents:
            role = content.get("role", "user")
            parts = content.get("parts", [])

            claude_role = "assistant" if role == "model" else "user"
            content_str: str | None = None
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResult] = []

            for part in parts:
                # Text
                if "text" in part and part["text"] is not None:
                    # Skip thought parts (thinking) - don't send back
                    if part.get("thought"):
                        continue
                    if part["text"]:
                        if content_str:
                            # Multiple text parts - need blocks
                            content_str = None
                            break
                        content_str = part["text"]

                # Function call → tool_use
                if "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=fc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            name=fc["name"],
                            arguments=fc.get("args", {}),
                        )
                    )

                # Function response → tool_result
                if "functionResponse" in part:
                    fr = part["functionResponse"]
                    resp = fr.get("response", {})
                    # Unwrap {result: ...} wrapper
                    if isinstance(resp, dict) and "result" in resp:
                        resp = resp["result"]
                    content_str_val = (
                        json.dumps(resp) if not isinstance(resp, str) else resp
                    )
                    tool_results.append(
                        ToolResult(
                            tool_call_id=fr.get("id", fr.get("name", "")),
                            content=content_str_val,
                        )
                    )

            # Build message
            msg = InternalMessage(role=claude_role)

            # Determine what to put in content vs tool fields
            has_tool_result = len(tool_results) > 0
            has_tool_call = len(tool_calls) > 0

            if has_tool_result:
                msg.tool_results = tool_results
                msg.tool_calls = []
                msg.content = content_str
            elif has_tool_call:
                msg.tool_calls = tool_calls
                msg.content = None
            else:
                msg.content = content_str

            if msg.content or msg.tool_calls or msg.tool_results:
                messages.append(msg)

        # --- Tools ---
        tools: list[ToolDef] = []
        if tools_raw:
            for tool_group in tools_raw:
                for func in tool_group.get("functionDeclarations", []):
                    raw_schema = func.get("parameters", {})
                    # Convert Gemini schema to JSON Schema
                    schema = _convert_schema(raw_schema)
                    # Claude requires top-level input_schema to be type "object"
                    schema["type"] = "object"
                    if "properties" not in schema:
                        schema["properties"] = {}
                    tools.append(
                        ToolDef(
                            name=func["name"],
                            description=func.get("description", ""),
                            parameters=schema,
                        )
                    )

        # --- Build request ---
        temperature = generation_config.get("temperature")
        max_tokens = generation_config.get("maxOutputTokens", 16384)

        return InternalRequest(
            model=model,
            messages=messages,
            system=system,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

    def format_request(self, req: InternalRequest) -> dict:
        """
        Format InternalRequest into Gemini/Antigravity request format.

        This is typically used when Gemini is the TARGET provider.
        """
        contents: list[dict] = []

        for msg in req.messages:
            parts: list[dict] = []

            # Content as text
            if msg.content:
                parts.append({"text": msg.content})

            # Tool calls as functionCall
            for tc in msg.tool_calls:
                parts.append(
                    {
                        "functionCall": {
                            "id": tc.id,
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    }
                )

            # Tool results as functionResponse
            for tr in msg.tool_results:
                parts.append(
                    {
                        "functionResponse": {
                            "id": tr.tool_call_id,
                            "response": {"result": tr.content},
                        }
                    }
                )

            if parts:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": parts})

        # System instruction
        system_instruction: dict | None = None
        if req.system:
            system_instruction = {"parts": [{"text": req.system}]}

        # Generation config
        generation_config: dict = {"maxOutputTokens": req.max_tokens}
        if req.temperature is not None:
            generation_config["temperature"] = req.temperature

        # Tools
        tools: list[dict] = []
        for tool in req.tools:
            tools.append(
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    ]
                }
            )

        return {
            "contents": contents,
            "systemInstruction": system_instruction,
            "generationConfig": generation_config,
            "tools": tools,
        }

    def parse_stream_event(
        self, event: dict, state: GeminiStreamState
    ) -> InternalStreamEvent | None:
        """
        Parse Gemini SSE event into InternalStreamEvent.

        This is used when Gemini is the TARGET provider.
        """
        # Gemini uses different format - this is for parsing Gemini responses
        # For now, return None as we primarily use this for source parsing
        return None

    def format_stream_event(
        self, event: InternalStreamEvent, state: GeminiStreamState
    ) -> str | None:
        """Format InternalStreamEvent into Gemini SSE format."""
        # Handle done event
        if event.type == "done":
            return None

        parts: list[dict] = []

        if event.type == "text" and event.text:
            parts.append({"text": event.text})

        if event.type == "thinking" and event.thinking:
            parts.append({"thought": True, "text": event.thinking})

        if event.type == "tool_call" and event.tool_call:
            tc = event.tool_call
            parts.append(
                {
                    "functionCall": {
                        "name": tc.name,
                        "args": tc.arguments,
                    }
                }
            )

        if not parts:
            return None

        candidate: dict = {
            "content": {
                "role": "model",
                "parts": parts,
            }
        }

        finish_reason_map = {
            "stop": "STOP",
            "max_tokens": "MAX_TOKENS",
            "tool_use": "STOP",
        }

        if event.finish_reason:
            candidate["finishReason"] = finish_reason_map.get(
                event.finish_reason, "STOP"
            )

        response: dict = {
            "candidates": [candidate],
            "modelVersion": state.model,
            "responseId": state.response_id,
        }

        if event.usage:
            response["usageMetadata"] = {
                "promptTokenCount": event.usage.input_tokens,
                "candidatesTokenCount": event.usage.output_tokens,
                "totalTokenCount": event.usage.total_tokens,
            }

        return json.dumps({"response": response})

    def get_headers(self, api_key: str) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def get_endpoint(self, base_url: str) -> str:
        return "/v1beta/models/{model}:streamGenerateContent"


# Gemini schema keys that are NOT valid in JSON Schema draft 2020-12
_GEMINI_ONLY_KEYS = {"nullable", "format", "title", "minimum", "maximum"}


def _convert_schema(schema: Any) -> dict:
    """Convert a Gemini parameter schema to JSON Schema draft 2020-12."""
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}

    result: dict = {}

    # Map type (Gemini uses uppercase sometimes)
    schema_type = schema.get("type", "object")
    if isinstance(schema_type, str):
        schema_type = schema_type.lower()
        type_map = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        schema_type = type_map.get(schema_type, "string")
    result["type"] = schema_type

    # Description
    if "description" in schema:
        result["description"] = schema["description"]

    # Enum
    if "enum" in schema:
        result["enum"] = schema["enum"]

    # Properties (recursive)
    if "properties" in schema:
        result["properties"] = {}
        for key, val in schema["properties"].items():
            result["properties"][key] = _convert_schema(val)

    # Required
    if "required" in schema:
        result["required"] = schema["required"]

    # Items (for arrays)
    if "items" in schema:
        result["items"] = _convert_schema(schema["items"])

    # anyOf / oneOf (recursive)
    for combo_key in ("anyOf", "oneOf"):
        if combo_key in schema:
            result[combo_key] = [_convert_schema(s) for s in schema[combo_key]]

    return result
