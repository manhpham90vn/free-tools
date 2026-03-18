"""
Gemini/Antigravity Adapter.

This adapter handles conversion between Google's Gemini API format and the internal
unified schema. The Gemini API is used by:
- Google's Antigravity Cloud Code extension
- Direct Gemini API calls

Gemini uses a unique format with "contents" (messages), "systemInstruction",
and "generationConfig" blocks. Tool calls use "functionCall" and "functionResponse".
"""

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
    """
    Gemini-specific stream state for tracking streaming events.

    Extends StreamState with Gemini-specific fields:
    - tool_call_accum: Accumulates tool call arguments across streaming chunks
    - in_thinking: Whether we're currently in a thinking block
    - usage: Token usage from the response
    """

    def __init__(self) -> None:
        super().__init__()
        # Maps index -> {id, name, arguments} for accumulating tool call data
        self.tool_call_accum: dict[int, dict] = {}
        self.in_thinking: bool = False
        self.usage: Usage | None = None


class GeminiAdapter(BaseAdapter):
    """
    Adapter for Gemini/Antigravity API format.

    Handles conversion between:
    - Gemini format: contents[], systemInstruction, generationConfig, tools
    - Internal format: messages[], system, temperature, tools
    """

    name = "gemini"

    def create_stream_state(self) -> GeminiStreamState:
        """Create a new GeminiStreamState for tracking streaming."""
        return GeminiStreamState()

    def parse_request(self, body: bytes, model: str) -> InternalRequest:
        """
        Parse Gemini/Antigravity request into InternalRequest.

        The Gemini API wraps requests in different ways depending on the client:
        - Direct Gemini API: {"contents": [...], "generationConfig": {...}}
        - Antigravity/Cloud Code: {"request": {...}} wrapper

        This method handles both formats and extracts:
        - System instruction
        - Messages (with role conversion: user/assistant)
        - Tools (function declarations)
        - Generation config (temperature, max_tokens)

        Args:
            body: Raw request body bytes (JSON)
            model: Model name from URL

        Returns:
            InternalRequest in unified format
        """
        raw = json.loads(body)

        # === Handle Antigravity envelope format ===
        # Antigravity wraps the real request in a "request" key
        inner = raw.get("request", raw)
        contents = inner.get("contents", [])
        system_instruction = inner.get("systemInstruction", {})
        generation_config = inner.get("generationConfig", {})
        tools_raw = inner.get("tools", [])

        # === Parse system instruction ===
        # Gemini uses systemInstruction.parts[].text for system prompts
        system_parts = []
        if system_instruction:
            for p in system_instruction.get("parts", []):
                text = p.get("text", "")
                if text:
                    system_parts.append(text)
        system = "\n".join(system_parts) if system_parts else None

        # === Parse messages ===
        # Gemini format: contents[{role, parts[{text|functionCall|functionResponse}]}]
        messages: list[InternalMessage] = []
        for content in contents:
            role = content.get("role", "user")
            parts = content.get("parts", [])

            # Convert Gemini roles to our internal roles
            # "model" in Gemini = "assistant" in internal schema
            claude_role = "assistant" if role == "model" else "user"
            content_str: str | None = None
            tool_calls: list[ToolCall] = []
            tool_results: list[ToolResult] = []

            for part in parts:
                # === Handle text content ===
                if "text" in part and part["text"] is not None:
                    # Skip "thought" parts - these are Claude's thinking, not content
                    if part.get("thought"):
                        continue
                    if part["text"]:
                        # If we already have content, need blocks format (not implemented)
                        if content_str:
                            content_str = None
                            break
                        content_str = part["text"]

                # === Handle function calls (tool calls) ===
                # Gemini uses "functionCall" for tool invocations
                if "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=fc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            name=fc["name"],
                            arguments=fc.get("args", {}),
                        )
                    )

                # === Handle function responses (tool results) ===
                # Gemini uses "functionResponse" for tool results
                if "functionResponse" in part:
                    fr = part["functionResponse"]
                    resp = fr.get("response", {})
                    # Unwrap {result: ...} wrapper if present
                    if isinstance(resp, dict) and "result" in resp:
                        resp = resp["result"]
                    # Convert to JSON string if not already string
                    content_str_val = (
                        json.dumps(resp) if not isinstance(resp, str) else resp
                    )
                    tool_results.append(
                        ToolResult(
                            tool_call_id=fr.get("id", fr.get("name", "")),
                            content=content_str_val,
                        )
                    )

            # === Build InternalMessage ===
            msg = InternalMessage(role=claude_role)

            # === Determine content vs tool fields ===
            # In internal schema, tool calls and tool results are separate fields
            has_tool_result = len(tool_results) > 0
            has_tool_call = len(tool_calls) > 0

            if has_tool_result:
                msg.tool_results = tool_results
                msg.tool_calls = []
                msg.content = content_str
            elif has_tool_call:
                msg.tool_calls = tool_calls
                msg.content = None  # Tool calls don't have text content
            else:
                msg.content = content_str

            # Only add non-empty messages
            if msg.content or msg.tool_calls or msg.tool_results:
                messages.append(msg)

        # === Parse tools ===
        # Gemini uses "tools" with "functionDeclarations"
        tools: list[ToolDef] = []
        if tools_raw:
            for tool_group in tools_raw:
                for func in tool_group.get("functionDeclarations", []):
                    raw_schema = func.get("parameters", {})
                    # Convert Gemini schema to JSON Schema (Gemini uses different format)
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

        # === Build InternalRequest ===
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

        This is typically used when Gemini is the TARGET provider
        (we're sending requests TO Gemini API).

        Converts:
        - InternalMessage[] -> contents[{role, parts[]}]
        - InternalMessage.tool_calls -> functionCall
        - InternalMessage.tool_results -> functionResponse
        - InternalRequest.system -> systemInstruction
        - InternalRequest.tools -> tools[functionDeclarations[]]

        Args:
            req: InternalRequest in unified format

        Returns:
            Dictionary in Gemini API format (will be JSON-serialized)
        """
        contents: list[dict] = []

        for msg in req.messages:
            parts: list[dict] = []

            # === Content as text part ===
            if msg.content:
                parts.append({"text": msg.content})

            # === Tool calls as functionCall ===
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

            # === Tool results as functionResponse ===
            for tr in msg.tool_results:
                parts.append(
                    {
                        "functionResponse": {
                            "id": tr.tool_call_id,
                            "response": {"result": tr.content},
                        }
                    }
                )

            # === Add message if it has any parts ===
            if parts:
                # Convert internal roles back to Gemini roles
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": parts})

        # === System instruction ===
        system_instruction: dict | None = None
        if req.system:
            system_instruction = {"parts": [{"text": req.system}]}

        # === Generation config ===
        generation_config: dict = {"maxOutputTokens": req.max_tokens}
        if req.temperature is not None:
            generation_config["temperature"] = req.temperature

        # === Tools ===
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
        We receive Gemini-format SSE and need to convert to internal format.

        Currently returns None because we primarily use this adapter
        for source parsing (formatting output for clients).

        Args:
            event: Raw event dict from Gemini SSE
            state: Stream state for accumulating context

        Returns:
            InternalStreamEvent or None to skip
        """
        # Gemini uses different format - this is for parsing Gemini responses
        # For now, return None as we primarily use this for source parsing
        return None

    def format_stream_event(
        self, event: InternalStreamEvent, state: GeminiStreamState
    ) -> str | None:
        """
        Format InternalStreamEvent into Gemini SSE format.

        This is used when Gemini is the SOURCE provider (client expects Gemini format).
        We convert internal events back to Gemini's SSE format.

        Converts:
        - InternalStreamEvent.text -> {"text": ...}
        - InternalStreamEvent.thinking -> {"thought": True, "text": ...}
        - InternalStreamEvent.tool_call -> {"functionCall": ...}
        - InternalStreamEvent.done -> triggers finishReason in response

        Args:
            event: InternalStreamEvent in unified format
            state: GeminiStreamState for tracking response ID, model, etc.

        Returns:
            SSE data line string (JSON), or None to skip
        """
        # Done events don't produce output
        if event.type == "done":
            return None

        parts: list[dict] = []

        # === Text content ===
        if event.type == "text" and event.text:
            parts.append({"text": event.text})

        # === Thinking (Claude's extended thinking) ===
        if event.type == "thinking" and event.thinking:
            parts.append({"thought": True, "text": event.thinking})

        # === Tool calls ===
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

        # Skip if no parts to send
        if not parts:
            return None

        # === Build Gemini response structure ===
        candidate: dict = {
            "content": {
                "role": "model",
                "parts": parts,
            }
        }

        # Map internal finish reasons to Gemini's finish reasons
        finish_reason_map = {
            "stop": "STOP",
            "max_tokens": "MAX_TOKENS",
            "tool_use": "STOP",
        }

        if event.finish_reason:
            candidate["finishReason"] = finish_reason_map.get(
                event.finish_reason, "STOP"
            )

        # Build full response with metadata
        response: dict = {
            "candidates": [candidate],
            "modelVersion": state.model,
            "responseId": state.response_id,
        }

        # Add usage metadata if available
        if event.usage:
            response["usageMetadata"] = {
                "promptTokenCount": event.usage.input_tokens,
                "candidatesTokenCount": event.usage.output_tokens,
                "totalTokenCount": event.usage.total_tokens,
            }

        return json.dumps({"response": response})

    def get_headers(self, api_key: str) -> dict[str, str]:
        """
        Get headers for Gemini API requests.

        Args:
            api_key: API key (not used for Gemini - uses different auth)

        Returns:
            Headers dict with Content-Type
        """
        return {"Content-Type": "application/json"}

    def get_endpoint(self, base_url: str) -> str:
        """
        Get the Gemini streaming endpoint path.

        Args:
            base_url: Base URL (ignored, returns fixed endpoint)

        Returns:
            Gemini streaming endpoint
        """
        return "/v1beta/models/{model}:streamGenerateContent"


# =============================================================================
# SCHEMA CONVERSION
# =============================================================================

# Gemini schema keys that are NOT valid in JSON Schema draft 2020-12
# These need to be filtered out when converting
_GEMINI_ONLY_KEYS = {"nullable", "format", "title", "minimum", "maximum"}


def _convert_schema(schema: Any) -> dict:
    """
    Convert a Gemini parameter schema to JSON Schema draft 2020-12.

    Gemini uses a slightly different schema format than standard JSON Schema.
    This function normalizes the schema for compatibility with other providers
    (especially Claude which requires valid JSON Schema).

    Handles:
    - Type mapping (Gemini sometimes uses uppercase)
    - Description, enum, properties, required, items
    - anyOf/oneOf combinations

    Args:
        schema: Gemini parameter schema dict

    Returns:
        JSON Schema dict
    """
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}

    result: dict = {}

    # === Map type (Gemini uses uppercase sometimes) ===
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

    # === Description ===
    if "description" in schema:
        result["description"] = schema["description"]

    # === Enum values ===
    if "enum" in schema:
        result["enum"] = schema["enum"]

    # === Properties (recursive conversion) ===
    if "properties" in schema:
        result["properties"] = {}
        for key, val in schema["properties"].items():
            result["properties"][key] = _convert_schema(val)

    # === Required fields ===
    if "required" in schema:
        result["required"] = schema["required"]

    # === Array items (recursive) ===
    if "items" in schema:
        result["items"] = _convert_schema(schema["items"])

    # === anyOf / oneOf (recursive) ===
    for combo_key in ("anyOf", "oneOf"):
        if combo_key in schema:
            result[combo_key] = [_convert_schema(s) for s in schema[combo_key]]

    return result
