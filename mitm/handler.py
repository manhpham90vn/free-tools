"""
Request interceptor for Antigravity MITM proxy.
Detects Gemini API calls, swaps models, converts format, and forwards to custom endpoint.
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


def should_intercept(path: str) -> bool:
    """Check if the request path matches an interceptable pattern."""
    return any(p.search(path) for p in INTERCEPT_PATTERNS)


def extract_model_from_path(path: str, body: bytes | None = None) -> Optional[str]:
    """Extract model name from URL path or request body.

    URL format: /v1beta/models/gemini-2.0-flash:generateContent
    Body format (Antigravity): {"model": "...", "request": {...}}
    """
    # Try URL first
    match = re.search(r"/models/([^/:]+)", path)
    if match:
        return match.group(1)

    # Try body for Antigravity format
    if body:
        try:
            import json

            data = json.loads(body)
            # Antigravity format: body.model or body.request.model
            model = data.get("model") or data.get("request", {}).get("model")
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

    Args:
        method: HTTP method
        path: Original request path
        headers: Original request headers
        body: Request body
        config: Application config dict

    Returns:
        Tuple of (status_code, response_headers, response_body)
    """
    # Get target from config or env
    target_url = config.get("target_endpoint") or os.environ.get("ANTHROPIC_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    model_mapping = config.get("model_mapping", {})

    if not target_url:
        raise ValueError(
            "No target endpoint configured (target_endpoint in config or ANTHROPIC_BASE_URL env)"
        )

    # Extract and swap model
    original_model = extract_model_from_path(path, body)
    default_model = config.get("default_model", "claude-sonnet-4-6")
    target_model = (
        swap_model(original_model, model_mapping, default_model)
        if original_model
        else default_model
    )

    print(f"[INTERCEPT] {original_model or 'unknown'} -> {target_model}")

    # Convert request format: Antigravity → Claude
    from .converter import convert_request

    try:
        claude_req = convert_request(body, target_model, config)
    except json.JSONDecodeError as e:
        return 400, {"Content-Type": "text/plain"}, f"Invalid JSON: {e}".encode()

    # Build headers for Anthropic API
    forward_headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key or "",
        "anthropic-version": "2023-06-01",
    }

    async with aiohttp.ClientSession() as session:
        async with session.request(
            method,
            target_url,
            headers=forward_headers,
            json=claude_req,
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

    Uses Anthropic SDK for proper SSE streaming.

    Args:
        method: HTTP method
        path: Original request path
        headers: Original request headers
        body: Request body
        config: Application config dict
        writer: asyncio StreamWriter to write response chunks to
    """
    from anthropic import AsyncAnthropic

    # Get target from config or env
    base_url = config.get("target_endpoint") or os.environ.get("ANTHROPIC_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    model_mapping = config.get("model_mapping", {})

    if not base_url:
        error = "No target endpoint configured"
        await send_error_response(writer, 500, "Internal Server Error", error)
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

    print(f"[STREAM] {original_model or 'unknown'} -> {target_model}")

    # Convert request format: Antigravity → Claude
    from .converter import convert_request, StreamState, convert_claude_event

    try:
        claude_req = convert_request(body, target_model, config)
    except (json.JSONDecodeError, Exception) as e:
        print(f"[ERROR] Convert request failed: {e}")
        import traceback

        traceback.print_exc()
        await send_error_response(writer, 400, "Bad Request", f"Convert error: {e}")
        return

    print(f"[STREAM] Target: {target_url}")
    print(f"[STREAM] Claude model: {claude_req.get('model')}")
    print(f"[STREAM] Messages: {len(claude_req.get('messages', []))}")
    print(f"[STREAM] Tools: {len(claude_req.get('tools', []))}")
    print(f"[STREAM] System: {str(claude_req.get('system', ''))[:100]}...")

    # Use Anthropic SDK for streaming (async)
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

        # Start streaming
        async with client.messages.stream(
            model=claude_req.get("model", target_model),
            messages=claude_req.get("messages", []),
            system=claude_req.get("system"),
            tools=claude_req.get("tools"),
            max_tokens=claude_req.get("max_tokens", 16384),
            temperature=claude_req.get("temperature"),
        ) as stream:
            # Send HTTP response headers
            status_line = "HTTP/1.1 200 OK\r\n"
            writer.write(status_line.encode())
            writer.write(b"Content-Type: text/event-stream\r\n")
            writer.write(b"Cache-Control: no-cache\r\n")
            writer.write(b"Connection: keep-alive\r\n")
            writer.write(b"Access-Control-Allow-Origin: *\r\n")
            writer.write(b"\r\n")
            await writer.drain()

            state = StreamState()
            state.model = target_model

            # Process stream events
            async for event in stream:
                event_type = event.type

                # Build event dict for converter
                event_dict = {"type": event_type}

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

                result = convert_claude_event(event_dict, state)
                if result:
                    try:
                        writer.write(f"data: {result}\r\n\r\n".encode())
                        await writer.drain()
                    except Exception as write_err:
                        print(f"[STREAM] Write error: {write_err}")
                        break

            # Send empty data to signal end
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
