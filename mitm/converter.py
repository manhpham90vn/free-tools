"""
Request/Response converter: Antigravity (Gemini Cloud Code) ↔ Anthropic Messages API

Based on 9router's translation chain:
  Request:  Antigravity envelope → Gemini → OpenAI → Claude
  Response: Claude SSE → OpenAI chunks → Antigravity SSE

Simplified here to direct conversion without OpenAI intermediate.
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# REQUEST: Antigravity → Claude
# ---------------------------------------------------------------------------


def convert_request(
    body: bytes,
    target_model: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert Antigravity (Gemini Cloud Code envelope) request to
    Anthropic Messages API format.
    """
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

    # --- Messages ---
    messages: List[Dict[str, Any]] = []
    for content in contents:
        role = content.get("role", "user")
        parts = content.get("parts", [])

        claude_role = "assistant" if role == "model" else "user"
        blocks: List[Dict[str, Any]] = []

        for part in parts:
            # Text
            if "text" in part and part["text"] is not None:
                # Skip thought parts (thinking) - don't send back to Claude
                if part.get("thought"):
                    continue
                if part["text"]:
                    blocks.append({"type": "text", "text": part["text"]})

            # Function call → tool_use
            if "functionCall" in part:
                fc = part["functionCall"]
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": fc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "name": fc["name"],
                        "input": fc.get("args", {}),
                    }
                )

            # Function response → tool_result
            if "functionResponse" in part:
                fr = part["functionResponse"]
                resp = fr.get("response", {})
                # Unwrap {result: ...} wrapper
                if isinstance(resp, dict) and "result" in resp:
                    resp = resp["result"]
                content_str = json.dumps(resp) if not isinstance(resp, str) else resp
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": fr.get("id", fr.get("name", "")),
                        "content": content_str,
                    }
                )

        if not blocks:
            continue

        # tool_result must be role=user
        has_tool_result = any(b["type"] == "tool_result" for b in blocks)
        has_tool_use = any(b["type"] == "tool_use" for b in blocks)

        if has_tool_result:
            tool_results = [b for b in blocks if b["type"] == "tool_result"]
            other = [b for b in blocks if b["type"] != "tool_result"]
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            if other:
                messages.append({"role": claude_role, "content": other})
        elif has_tool_use:
            messages.append({"role": "assistant", "content": blocks})
        else:
            # Simple text - use string if single text block
            if len(blocks) == 1 and blocks[0]["type"] == "text":
                messages.append({"role": claude_role, "content": blocks[0]["text"]})
            else:
                messages.append({"role": claude_role, "content": blocks})

    # --- Build Claude request ---
    claude_req: Dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": True,
    }

    if system_parts:
        claude_req["system"] = "\n".join(system_parts)

    if "temperature" in generation_config:
        claude_req["temperature"] = generation_config["temperature"]

    max_tokens = generation_config.get("maxOutputTokens", 16384)
    claude_req["max_tokens"] = max_tokens

    # --- Tools ---
    if tools_raw:
        claude_tools = []
        for tool_group in tools_raw:
            for func in tool_group.get("functionDeclarations", []):
                raw_schema = func.get("parameters", {})
                # Convert Gemini schema to Claude-compatible JSON Schema draft 2020-12
                schema = _convert_schema(raw_schema)
                # Claude requires top-level input_schema to be type "object"
                schema["type"] = "object"
                if "properties" not in schema:
                    schema["properties"] = {}
                claude_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": schema,
                    }
                )
        if claude_tools:
            claude_req["tools"] = claude_tools
            print(f"[TOOLS] Converted {len(claude_tools)} tools to Claude format")

    return claude_req


# Gemini schema keys that are NOT valid in JSON Schema draft 2020-12
_GEMINI_ONLY_KEYS = {"nullable", "format", "title", "minimum", "maximum"}


def _convert_schema(schema: Any) -> Dict[str, Any]:
    """Convert a Gemini parameter schema to JSON Schema draft 2020-12 for Claude.

    Gemini uses a subset of OpenAPI 3.0 schema which differs from JSON Schema:
    - Uses "nullable: true" instead of anyOf with null
    - May have non-standard keys
    - Top-level must be type "object" for Claude
    """
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}

    result: Dict[str, Any] = {}

    # Map type (Gemini uses uppercase sometimes)
    schema_type = schema.get("type", "object")
    if isinstance(schema_type, str):
        schema_type = schema_type.lower()
        # Gemini type mappings
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


# ---------------------------------------------------------------------------
# RESPONSE: Claude SSE → Antigravity SSE
# ---------------------------------------------------------------------------


class StreamState:
    """Tracks state across streaming events."""

    __slots__ = (
        "message_id",
        "model",
        "tool_call_accum",
        "response_id",
        "in_thinking",
        "usage",
    )

    def __init__(self) -> None:
        self.message_id: str = ""
        self.model: str = ""
        self.tool_call_accum: Dict[int, Dict] = {}  # index → {id, name, arguments}
        self.response_id: str = f"resp_{uuid.uuid4().hex[:12]}"
        self.in_thinking: bool = False
        self.usage: Optional[Dict] = None


def convert_claude_event(
    event_data: Dict[str, Any], state: StreamState
) -> Optional[str]:
    """
    Convert a single Claude SSE event to Antigravity SSE line.

    Returns SSE data line (without "data: " prefix) or None to skip.
    """
    event_type = event_data.get("type", "")

    if event_type == "message_start":
        msg = event_data.get("message", {})
        state.message_id = msg.get("id", f"msg_{int(time.time())}")
        state.model = msg.get("model", "")
        return None  # No output for message_start

    elif event_type == "content_block_start":
        block = event_data.get("content_block", {})
        block_type = block.get("type", "")
        index = event_data.get("index", 0)

        if block_type == "thinking":
            state.in_thinking = True
            return None

        elif block_type == "tool_use":
            state.tool_call_accum[index] = {
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": "",
            }
            return None  # Accumulate, don't emit yet

        elif block_type == "text":
            return None  # Wait for deltas

    elif event_type == "content_block_delta":
        delta = event_data.get("delta", {})
        delta_type = delta.get("type", "")
        index = event_data.get("index", 0)

        if delta_type == "text_delta" and delta.get("text"):
            parts = [{"text": delta["text"]}]
            return _build_antigravity_response(parts, state)

        elif delta_type == "thinking_delta" and delta.get("thinking"):
            parts = [{"thought": True, "text": delta["thinking"]}]
            return _build_antigravity_response(parts, state)

        elif delta_type == "input_json_delta" and delta.get("partial_json"):
            if index in state.tool_call_accum:
                state.tool_call_accum[index]["arguments"] += delta["partial_json"]
            return None  # Accumulate

    elif event_type == "content_block_stop":
        state.in_thinking = False
        return None

    elif event_type == "message_delta":
        # Extract usage
        usage = event_data.get("usage", {})
        if usage:
            input_tokens = usage.get("input_tokens") or 0
            output_tokens = usage.get("output_tokens") or 0
            cache_read = usage.get("cache_read_input_tokens") or 0
            cache_creation = usage.get("cache_creation_input_tokens") or 0
            state.usage = {
                "promptTokenCount": input_tokens + cache_read + cache_creation,
                "candidatesTokenCount": output_tokens,
                "totalTokenCount": input_tokens
                + output_tokens
                + cache_read
                + cache_creation,
            }

        stop_reason = event_data.get("delta", {}).get("stop_reason")
        if stop_reason:
            # Emit accumulated tool calls
            parts = []
            for idx in sorted(state.tool_call_accum.keys()):
                accum = state.tool_call_accum[idx]
                args = {}
                try:
                    args = json.loads(accum["arguments"])
                except (json.JSONDecodeError, ValueError):
                    pass
                parts.append(
                    {
                        "functionCall": {
                            "name": accum["name"],
                            "args": args,
                        }
                    }
                )

            # Add empty text if no parts
            if not parts:
                parts.append({"text": ""})

            # Map stop reason
            reason_map = {
                "end_turn": "STOP",
                "max_tokens": "MAX_TOKENS",
                "tool_use": "STOP",
                "stop_sequence": "STOP",
            }
            finish_reason = reason_map.get(stop_reason, "STOP")

            return _build_antigravity_response(
                parts,
                state,
                finish_reason=finish_reason,
                include_usage=True,
            )

    elif event_type == "message_stop":
        return None

    return None


def _build_antigravity_response(
    parts: List[Dict],
    state: StreamState,
    finish_reason: Optional[str] = None,
    include_usage: bool = False,
) -> str:
    """Build Antigravity SSE response JSON string."""
    candidate: Dict[str, Any] = {
        "content": {
            "role": "model",
            "parts": parts,
        }
    }

    if finish_reason:
        candidate["finishReason"] = finish_reason

    response: Dict[str, Any] = {
        "candidates": [candidate],
        "modelVersion": state.model,
        "responseId": state.response_id,
    }

    if include_usage and state.usage:
        response["usageMetadata"] = state.usage

    return json.dumps({"response": response})
