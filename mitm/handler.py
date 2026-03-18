"""
Request interceptor for Antigravity MITM proxy.
Detects API calls, converts format using providers module, and forwards to target endpoint.
"""

import json
import os
import re
from typing import Optional, Dict, Any

import aiohttp


# URL patterns that indicate a Gemini chat/generate request
INTERCEPT_PATTERNS = [
    re.compile(r":generateContent"),
    re.compile(r":streamGenerateContent"),
]

# OpenAI patterns
OPENAI_PATTERNS = [
    re.compile(r"/v1/chat/completions"),
    re.compile(r"/v1/completions"),
]

# Claude patterns (direct Claude API)
CLAUDE_PATTERNS = [
    re.compile(r"/v1/messages"),
]


def should_intercept(path: str) -> bool:
    """Check if the request path matches an interceptable pattern."""
    return any(p.search(path) for p in INTERCEPT_PATTERNS)


def detect_provider(path: str, headers: Dict[str, str] | None = None) -> str:
    """
    Detect the source provider from request path.

    Returns:
        Provider name: "gemini", "openai", "claude"
    """
    # Check for Gemini/Antigravity
    if any(p.search(path) for p in INTERCEPT_PATTERNS):
        return "gemini"

    # Check for OpenAI
    if any(p.search(path) for p in OPENAI_PATTERNS):
        return "openai"

    # Check for Claude direct API
    if any(p.search(path) for p in CLAUDE_PATTERNS):
        return "claude"

    # Default to Gemini (Antigravity)
    return "gemini"


def extract_model_from_path(path: str, body: bytes | None = None) -> Optional[str]:
    """Extract model name from URL path or request body.

    URL format: /v1beta/models/gemini-2.0-flash:generateContent
    Body format (Antigravity): {"model": "...", "request": {...}}
    """
    # Try URL first - Gemini format
    match = re.search(r"/models/([^/:]+)", path)
    if match:
        return match.group(1)

    # Try URL - OpenAI format
    match = re.search(r"/v1/chat/completions/(.+)$", path)
    if match:
        return match.group(1)

    # Try body for Antigravity format
    if body:
        try:
            data = json.loads(body)
            # Antigravity format: body.model or body.request.model
            model = data.get("model") or data.get("request", {}).get("model")
            if model:
                return model
            # OpenAI format
            model = data.get("model")
            if model:
                return model
        except Exception:
            pass

    return None


def swap_model(
    model: str, model_mapping: Dict[str, str], default: str = "claude-sonnet-4-6"
) -> str:
    """Swap model name according to mapping config, with default fallback."""
    return model_mapping.get(model, default)


async def forward_to_target(
    method: str,
    path: str,
    headers: Dict[str, str],
    body: bytes,
    config: Dict[str, Any],
) -> tuple[int, Dict[str, str], bytes]:
    """
    Forward an intercepted request to the custom target endpoint.

    Uses the providers module for conversion.
    """
    from providers import get_adapter

    # Detect source provider
    source_provider = detect_provider(path, headers)

    # Get target provider from config
    target_provider_name = config.get("target_provider", "claude")

    # Get adapters
    source_adapter = get_adapter(source_provider)
    target_adapter = get_adapter(target_provider_name)

    # Get config
    base_url = config.get("target_endpoint") or os.environ.get("ANTHROPIC_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("ANTHROPIC_AUTH_TOKEN")

    if not base_url:
        raise ValueError(
            "No target endpoint configured (target_endpoint in config or ANTHROPIC_BASE_URL env)"
        )

    # Extract and swap model
    original_model = extract_model_from_path(path, body)
    default_model = config.get("default_model", "claude-sonnet-4-6")
    model_mapping = config.get("model_mapping", {})
    target_model = (
        swap_model(original_model, model_mapping, default_model)
        if original_model
        else default_model
    )

    print(
        f"[INTERCEPT] {original_model or 'unknown'} -> {target_model} ({source_provider} -> {target_provider_name})"
    )

    try:
        # Parse source request to internal format
        internal_req = source_adapter.parse_request(body, target_model)
        # Override model with swapped model
        internal_req.model = target_model
    except json.JSONDecodeError as e:
        return 400, {"Content-Type": "text/plain"}, f"Invalid JSON: {e}".encode()

    # Format for target provider
    target_req = target_adapter.format_request(internal_req)

    # Build headers for target
    forward_headers = target_adapter.get_headers(api_key or "")

    # Get endpoint
    endpoint = target_adapter.get_endpoint(base_url)
    target_url = f"{base_url.rstrip('/')}{endpoint}"

    async with aiohttp.ClientSession() as session:
        async with session.request(
            method,
            target_url,
            headers=forward_headers,
            json=target_req,
            ssl=True,
        ) as resp:
            resp_body = await resp.read()
            resp_headers = dict(resp.headers)
            return resp.status, resp_headers, resp_body


async def forward_to_target_streaming(
    method: str,
    path: str,
    headers: Dict[str, str],
    body: bytes,
    config: Dict[str, Any],
    writer: Any,
) -> None:
    """
    Forward an intercepted request and stream the response back.

    Uses the providers module for conversion.
    """
    from providers import get_adapter

    # Detect source provider
    source_provider = detect_provider(path, headers)

    # Get target provider from config
    target_provider_name = config.get("target_provider", "claude")

    # Get adapters
    source_adapter = get_adapter(source_provider)
    target_adapter = get_adapter(target_provider_name)

    # Get config
    base_url = config.get("target_endpoint") or os.environ.get("ANTHROPIC_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    model_mapping = config.get("model_mapping", {})

    if not base_url:
        await send_error_response(
            writer, 500, "Internal Server Error", "No target endpoint configured"
        )
        return

    # Ensure base URL doesn't include /v1/messages (SDK appends it)
    target_url = base_url.rstrip("/")
    if target_url.endswith("/v1/messages"):
        target_url = target_url[: -len("/v1/messages")]

    # Extract and swap model
    original_model = extract_model_from_path(path, body)
    default_model = config.get("default_model", "claude-sonnet-4-6")
    target_model = (
        swap_model(original_model, model_mapping, default_model)
        if original_model
        else default_model
    )

    print(
        f"[STREAM] {original_model or 'unknown'} -> {target_model} ({source_provider} -> {target_provider_name})"
    )

    try:
        # Parse source request to internal format
        internal_req = source_adapter.parse_request(body, target_model)
        # Override model with swapped model
        internal_req.model = target_model
    except (json.JSONDecodeError, Exception) as e:
        print(f"[ERROR] Convert request failed: {e}")
        import traceback

        traceback.print_exc()
        await send_error_response(writer, 400, "Bad Request", f"Convert error: {e}")
        return

    # Format for target provider
    target_req = target_adapter.format_request(internal_req)

    print(f"[STREAM] Target: {target_url}")
    print(f"[STREAM] Model: {target_req.get('model')}")
    print(f"[STREAM] Messages: {len(target_req.get('messages', []))}")

    # Special handling for Claude target (use SDK)
    if target_provider_name == "claude":
        await _stream_claude(
            target_url, api_key or "", target_req, source_adapter, writer
        )
    else:
        # Generic streaming for other providers
        await _stream_generic(
            target_url,
            api_key or "",
            target_req,
            target_adapter,
            source_adapter,
            writer,
        )


async def _stream_claude(
    target_url: str,
    api_key: str,
    target_req: dict,
    source_adapter: Any,
    writer: Any,
) -> None:
    """Stream using Anthropic SDK."""
    from anthropic import AsyncAnthropic

    try:
        import httpx

        async def _override_user_agent(request: httpx.Request) -> None:
            request.headers["user-agent"] = "antigravity/1.20.5 linux/amd64"

        http_client = httpx.AsyncClient(
            event_hooks={"request": [_override_user_agent]},
            timeout=httpx.Timeout(300.0),
        )
        client = AsyncAnthropic(
            http_client=http_client,
            base_url=target_url,
            api_key=api_key,
        )

        # Build stream kwargs
        stream_kwargs: dict[str, Any] = {
            "model": target_req.get("model", "claude-sonnet-4-6"),
            "messages": target_req.get("messages", []),
            "max_tokens": target_req.get("max_tokens", 16384),
        }
        if target_req.get("system") is not None:
            stream_kwargs["system"] = target_req["system"]
        if target_req.get("tools") is not None:
            stream_kwargs["tools"] = target_req["tools"]
        if target_req.get("temperature") is not None:
            stream_kwargs["temperature"] = target_req["temperature"]

        # Start streaming
        print(f"[STREAM] Connecting to {target_url}...")
        async with client.messages.stream(**stream_kwargs) as stream:
            print("[STREAM] Connected, sending response headers...")
            # Send HTTP response headers
            status_line = "HTTP/1.1 200 OK\r\n"
            writer.write(status_line.encode())
            writer.write(b"Content-Type: text/event-stream\r\n")
            writer.write(b"Cache-Control: no-cache\r\n")
            writer.write(b"Connection: keep-alive\r\n")
            writer.write(b"Access-Control-Allow-Origin: *\r\n")
            writer.write(b"\r\n")
            await writer.drain()

            # Create source and target states
            source_state = source_adapter.create_stream_state()
            target_state = source_adapter.create_stream_state()  # ClaudeStreamState

            # Process stream events
            event_count = 0
            async for event in stream:
                event_count += 1
                event_type = event.type
                if event_count <= 3:
                    print(f"[STREAM] Event #{event_count}: {event_type}")

                # Build event dict for adapter
                event_dict: dict[str, Any] = {"type": event_type}

                if hasattr(event, "message"):
                    event_dict["message"] = event.message.model_dump()

                if hasattr(event, "content_block"):
                    event_dict["content_block"] = event.content_block.model_dump()

                if hasattr(event, "delta"):
                    event_dict["delta"] = event.delta.model_dump()

                if hasattr(event, "index"):
                    event_dict["index"] = event.index

                if hasattr(event, "usage"):
                    event_dict["usage"] = event.usage.model_dump()

                # Parse target event to internal
                internal_event = _parse_claude_event(event_dict, target_state)
                if internal_event is None:
                    continue

                # Format for source
                result = source_adapter.format_stream_event(
                    internal_event, source_state
                )
                if result:
                    try:
                        writer.write(f"data: {result}\r\n\r\n".encode())
                        await writer.drain()
                    except Exception as write_err:
                        print(f"[STREAM] Write error: {write_err}")
                        break

            # Send empty data to signal end
            print(f"[STREAM] Done, processed {event_count} events")
            try:
                writer.write(b"data: \r\n\r\n")
                await writer.drain()
            except Exception:
                pass

    except Exception as e:
        print(f"[ERROR] Streaming failed: {e}")
        import traceback

        traceback.print_exc()
        await send_error_response(writer, 500, "Internal Server Error", str(e))
    finally:
        try:
            writer.close()
        except Exception:
            pass


def _parse_claude_event(event: dict, state: Any) -> Any:
    """Parse Claude event to InternalStreamEvent."""
    from providers.schema import InternalStreamEvent, ToolCall, Usage

    event_type = event.get("type", "")

    if event_type == "message_start":
        msg = event.get("message", {})
        state.message_id = msg.get("id", "")
        state.model = msg.get("model", "")
        return None

    elif event_type == "content_block_start":
        block = event.get("content_block", {})
        block_type = block.get("type", "")
        index = event.get("index", 0)

        if block_type == "tool_use":
            state.tool_call_accum[index] = {
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": "",
            }
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

    elif event_type == "message_delta":
        usage = event.get("usage", {})
        if usage:
            input_tokens = usage.get("input_tokens") or 0
            output_tokens = usage.get("output_tokens") or 0
            state.usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
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
                        id=accum["id"], name=accum["name"], arguments=args
                    ),
                )

        if stop_reason:
            return InternalStreamEvent(
                type="done",
                finish_reason=stop_reason,
                usage=state.usage,
            )

    return None


async def _stream_generic(
    target_url: str,
    api_key: str,
    target_req: dict,
    target_adapter: Any,
    source_adapter: Any,
    writer: Any,
) -> None:
    """Generic streaming for non-Claude targets (OpenAI, etc.)."""
    headers = target_adapter.get_headers(api_key or "")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                target_url,
                headers=headers,
                json=target_req,
            ) as resp:
                # Send HTTP response headers
                status_line = "HTTP/1.1 200 OK\r\n"
                writer.write(status_line.encode())
                writer.write(b"Content-Type: text/event-stream\r\n")
                writer.write(b"Cache-Control: no-cache\r\n")
                writer.write(b"Connection: keep-alive\r\n")
                writer.write(b"Access-Control-Allow-Origin: *\r\n")
                writer.write(b"\r\n")
                await writer.drain()

                # Create states
                source_state = source_adapter.create_stream_state()
                target_state = target_adapter.create_stream_state()

                # Read SSE stream
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: "):
                        continue

                    data = line_str[6:]  # Remove "data: "
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # Parse target event to internal
                    internal_event = target_adapter.parse_stream_event(
                        event, target_state
                    )
                    if internal_event is None:
                        continue

                    # Format for source
                    result = source_adapter.format_stream_event(
                        internal_event, source_state
                    )
                    if result:
                        try:
                            writer.write(f"data: {result}\r\n\r\n".encode())
                            await writer.drain()
                        except Exception as write_err:
                            print(f"[STREAM] Write error: {write_err}")
                            break

                # Send end signal
                try:
                    writer.write(b"data: \r\n\r\n")
                    await writer.drain()
                except Exception:
                    pass

    except Exception as e:
        print(f"[ERROR] Generic streaming failed: {e}")
        import traceback

        traceback.print_exc()
        await send_error_response(writer, 500, "Internal Server Error", str(e))
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def send_error_response(
    writer: Any,
    status_code: int,
    reason: str,
    message: str,
) -> None:
    """Send an error response to the client."""
    body = f"""<!DOCTYPE html>
<html><body>
<h1>{status_code} {reason}</h1>
<p>{message}</p>
</body></html>"""

    status_line = f"HTTP/1.1 {status_code} {reason}\r\n"
    writer.write(status_line.encode())
    writer.write(b"Content-Type: text/html\r\n")
    writer.write(f"Content-Length: {len(body)}\r\n".encode())
    writer.write(b"Connection: close\r\n")
    writer.write(b"\r\n")
    writer.write(body.encode())
    await writer.drain()
    writer.close()
