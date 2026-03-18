"""
Request Interceptor for Antigravity MITM Proxy.

This module contains the core request handling logic:
- Provider detection: Identifies whether the request is from Gemini, OpenAI, or Claude
- Request conversion: Translates between different API formats using the providers module
- Response forwarding: Sends converted requests to the target LLM provider
- Streaming support: Handles Server-Sent Events (SSE) for real-time responses
- Model mapping: Swaps source model names to target model names based on config

The module uses the providers/ module as an abstraction layer.
Each provider (Gemini, OpenAI, Claude) has an adapter that handles format conversion.
"""

# === Standard library imports ===
import json  # JSON parsing for request/response bodies
import os  # Environment variables for API keys
import re  # Regular expressions for URL pattern matching
from typing import Optional, Dict, Any  # Type hints

# === Third-party imports ===
import aiohttp  # Async HTTP client for forwarding requests to target LLM

# === Internal module imports ===
from .utils import send_error_response  # Utility for sending error responses to clients


# =============================================================================
# URL PATTERNS FOR PROVIDER DETECTION
# =============================================================================

# These patterns are used to detect which LLM API the request is targeting.
# The interceptor uses these to determine:
# 1. Whether to intercept the request at all
# 2. Which provider adapter to use for format conversion

# Gemini/Antigravity patterns (Google's AI SDK)
# :generateContent = single request, :streamGenerateContent = streaming
INTERCEPT_PATTERNS = [
    re.compile(r":generateContent"),  # Non-streaming Gemini API
    re.compile(r":streamGenerateContent"),  # Streaming Gemini API
]

# OpenAI patterns (standard OpenAI API format)
OPENAI_PATTERNS = [
    re.compile(r"/v1/chat/completions"),  # Chat Completions API
    re.compile(r"/v1/completions"),  # Legacy Completions API
]

# Claude direct API patterns (Anthropic's API)
CLAUDE_PATTERNS = [
    re.compile(r"/v1/messages"),  # Claude Messages API
]


# =============================================================================
# PROVIDER DETECTION
# =============================================================================


def should_intercept(path: str) -> bool:
    """
    Check if the request path matches an interceptable pattern.

    Only requests matching these patterns (Gemini API endpoints) will be
    intercepted and forwarded to the configured target provider.
    Other requests (like loading web UI, fetching user info) will be
    passed through to the real server.

    Args:
        path: The URL path from the HTTP request (e.g., "/v1beta/models/gemini-pro:generateContent")

    Returns:
        True if this request should be intercepted, False for passthrough
    """
    return any(p.search(path) for p in INTERCEPT_PATTERNS)


def detect_provider(path: str, headers: Dict[str, str] | None = None) -> str:
    """
    Detect the source provider from the request URL path.

    This determines which format the incoming request is using:
    - "gemini": Google's Gemini API format (Antigravity)
    - "openai": OpenAI's Chat Completions format
    - "claude": Anthropic's Claude Messages API format

    The detected provider is used to select the appropriate adapter
    for converting the request to our internal format.

    Args:
        path: URL path from the HTTP request
        headers: Optional HTTP headers (not currently used but available for future detection)

    Returns:
        Provider identifier string: "gemini", "openai", or "claude"
    """
    # Check for Gemini/Antigravity endpoints first (most common use case)
    if any(p.search(path) for p in INTERCEPT_PATTERNS):
        return "gemini"

    # Check for OpenAI format
    if any(p.search(path) for p in OPENAI_PATTERNS):
        return "openai"

    # Check for Claude direct API
    if any(p.search(path) for p in CLAUDE_PATTERNS):
        return "claude"

    # Default to Gemini if no pattern matches
    # This is because Antigravity primarily targets Google's Gemini API
    return "gemini"


# =============================================================================
# MODEL EXTRACTION AND MAPPING
# =============================================================================


def extract_model_from_path(path: str, body: bytes | None = None) -> Optional[str]:
    """
    Extract the model name from either the URL path or request body.

    Different API providers put the model name in different places:
    - Gemini: /v1/models/gemini-2.0-flash:generateContent (in URL path)
    - OpenAI: /v1/chat/completions (model in body as {"model": "gpt-4"})
    - Antigravity: {"model": "...", "request": {...}} (wrapper format)

    This function tries each location until it finds a model name.

    Args:
        path: URL path from the HTTP request
        body: Optional request body as bytes (for JSON parsing)

    Returns:
        The extracted model name string, or None if not found
    """
    # Try URL path first - Gemini format
    # Pattern: /v1/models/{model-name}:generateContent
    match = re.search(r"/models/([^/:]+)", path)
    if match:
        return match.group(1)

    # Try URL - OpenAI format
    # Pattern: /v1/chat/completions/{model-name}
    match = re.search(r"/v1/chat/completions/(.+)$", path)
    if match:
        return match.group(1)

    # Try parsing body for Antigravity/OpenAI format
    # The body might be JSON with {"model": "..."} or {"request": {"model": "..."}}
    if body:
        try:
            data = json.loads(body)
            # Antigravity wrapper format: body.model or body.request.model
            model = data.get("model") or data.get("request", {}).get("model")
            if model:
                return model
        except Exception:
            # JSON parsing failed, ignore
            pass

    return None


def swap_model(
    model: str, model_mapping: Dict[str, str], default: str = "claude-sonnet-4-6"
) -> str:
    """
    Swap the model name according to the configuration mapping.

    This allows users to map incoming model names to different target models.
    For example:
    - gemini-2.5-flash -> claude-opus-4-6
    - gpt-4 -> claude-sonnet-4-6

    If no mapping exists, the default model is used.

    Args:
        model: The original model name from the incoming request
        model_mapping: Dictionary mapping original -> target model names
        default: Fallback model name if no mapping exists

    Returns:
        The mapped (or default) model name
    """
    return model_mapping.get(model, default)


# =============================================================================
# REQUEST FORWARDING (NON-STREAMING)
# =============================================================================


async def forward_to_target(
    method: str,
    path: str,
    headers: Dict[str, str],
    body: bytes,
    config: Dict[str, Any],
) -> tuple[int, Dict[str, str], bytes]:
    """
    Forward an intercepted request to the custom target endpoint.

    This is the main entry point for non-streaming requests. It:
    1. Detects the source provider from the URL
    2. Gets the target provider from config
    3. Extracts and maps the model name
    4. Parses the source request into internal format
    5. Converts to target provider's format
    6. Sends the request to the target LLM
    7. Returns the response (status, headers, body)

    Args:
        method: HTTP method (GET, POST, etc.)
        path: URL path from the request
        headers: HTTP headers from the request
        body: Request body as bytes
        config: Application configuration dictionary

    Returns:
        Tuple of (HTTP status code, response headers dict, response body bytes)
    """
    # Import here to avoid circular imports
    from providers import get_adapter

    # Step 1: Detect source provider (gemini, openai, claude)
    source_provider = detect_provider(path, headers)

    # Step 2: Get target provider from config (default: claude)
    target_provider_name = config.get("target_provider", "claude")

    # Step 3: Get the adapter instances for source and target
    # Adapters handle format conversion between providers
    source_adapter = get_adapter(source_provider)
    target_adapter = get_adapter(target_provider_name)

    # Step 4: Get endpoint and API key from environment variables (required)
    # These MUST be set in .env file - no fallback to config
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")

    if not base_url:
        raise ValueError(
            "ANTHROPIC_BASE_URL not set in environment (.env file required)"
        )
    if not api_key:
        raise ValueError(
            "ANTHROPIC_AUTH_TOKEN not set in environment (.env file required)"
        )

    # Step 5: Extract and swap model name
    original_model = extract_model_from_path(path, body)
    default_model = config.get("default_model", "claude-sonnet-4-6")
    model_mapping = config.get("model_mapping", {})
    # Map the original model to target model (or use default)
    target_model = (
        swap_model(original_model, model_mapping, default_model)
        if original_model
        else default_model
    )

    # Log the conversion for debugging
    print(
        f"[INTERCEPT] {original_model or 'unknown'} -> {target_model} ({source_provider} -> {target_provider_name})"
    )

    # Step 6: Parse source request to internal format
    # The source adapter converts from provider format to our internal schema
    try:
        internal_req = source_adapter.parse_request(body, target_model)
        # Override the model with the mapped target model
        internal_req.model = target_model
    except json.JSONDecodeError as e:
        # Return 400 Bad Request if the request body is invalid JSON
        return 400, {"Content-Type": "text/plain"}, f"Invalid JSON: {e}".encode()

    # Step 7: Format for target provider
    # The target adapter converts from internal schema to provider format
    target_req = target_adapter.format_request(internal_req)

    # Step 8: Build headers for target request
    # Each provider has different auth header requirements
    forward_headers = target_adapter.get_headers(api_key or "")

    # Step 9: Get the target endpoint path
    endpoint = target_adapter.get_endpoint(base_url)
    target_url = f"{base_url.rstrip('/')}{endpoint}"

    # Step 10: Send request to target LLM
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method,
            target_url,
            headers=forward_headers,
            json=target_req,
            ssl=True,  # Verify SSL certificates for security
        ) as resp:
            # Step 11: Read and return the response
            resp_body = await resp.read()
            resp_headers = dict(resp.headers)
            return resp.status, resp_headers, resp_body


# =============================================================================
# REQUEST FORWARDING (STREAMING)
# =============================================================================


async def forward_to_target_streaming(
    method: str,
    path: str,
    headers: Dict[str, str],
    body: bytes,
    config: Dict[str, Any],
    writer: Any,
) -> None:
    """
    Forward an intercepted streaming request to the target provider.

    Streaming requests use Server-Sent Events (SSE) to stream responses
    in real-time. This function:
    1. Detects provider and extracts model
    2. Converts request format
    3. Forwards to target LLM with streaming enabled
    4. Converts each streaming event back to source format
    5. Streams the converted events back to the original client

    The writer is an asyncio StreamWriter that we write SSE data to.

    Args:
        method: HTTP method (should be POST for streaming)
        path: URL path from the request
        headers: HTTP headers
        body: Request body as bytes
        config: Application configuration
        writer: StreamWriter to send response back to client
    """
    from providers import get_adapter

    # Same provider detection as non-streaming
    source_provider = detect_provider(path, headers)
    target_provider_name = config.get("target_provider", "claude")

    source_adapter = get_adapter(source_provider)
    target_adapter = get_adapter(target_provider_name)

    # Get endpoint and API key from environment variables (required)
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    model_mapping = config.get("model_mapping", {})

    # Validate target endpoint
    if not base_url:
        await send_error_response(
            writer, 500, "Internal Server Error", "ANTHROPIC_BASE_URL not set in .env"
        )
        return
    if not api_key:
        await send_error_response(
            writer, 500, "Internal Server Error", "ANTHROPIC_AUTH_TOKEN not set in .env"
        )
        return

    # Fix: Remove /v1/messages from base URL if present
    # The SDK/httpx appends this automatically, so it would be duplicated
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

    # Parse and convert request
    try:
        internal_req = source_adapter.parse_request(body, target_model)
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

    # Special handling for Claude streaming (uses Anthropic SDK)
    if target_provider_name == "claude":
        await _stream_claude(
            target_url, api_key or "", target_req, source_adapter, writer
        )
    else:
        # Generic streaming for other providers (OpenAI, etc.)
        await _stream_generic(
            target_url,
            api_key or "",
            target_req,
            target_adapter,
            source_adapter,
            writer,
        )


# =============================================================================
# CLAUDE STREAMING HANDLER (uses Anthropic SDK)
# =============================================================================


async def _stream_claude(
    target_url: str,
    api_key: str,
    target_req: dict,
    source_adapter: Any,
    writer: Any,
) -> None:
    """
    Handle streaming for Claude API using the Anthropic SDK.

    The Anthropic SDK provides a convenient streaming interface.
    We:
    1. Create an AsyncAnthropic client with custom HTTP client
    2. Override User-Agent to hide that we're using the SDK
    3. Start streaming with the messages API
    4. Convert each Claude event to the source provider's format
    5. Stream the converted events back to the client

    The source_adapter handles converting Claude's event format
    to whatever format the original client expects (Gemini SSE).

    Args:
        target_url: Base URL for Anthropic API
        api_key: Anthropic API key
        target_req: Formatted request dict
        source_adapter: Adapter to convert events back to source format
        writer: StreamWriter to send response to client
    """
    from anthropic import AsyncAnthropic

    try:
        # Create custom httpx client to override User-Agent
        # Some servers reject requests with python-httpx user agent
        import httpx

        async def _override_user_agent(request: httpx.Request) -> None:
            request.headers["user-agent"] = "antigravity/1.20.5 linux/amd64"

        # Configure HTTP client with custom user-agent and long timeout
        # Claude can take a while to generate long responses
        http_client = httpx.AsyncClient(
            event_hooks={"request": [_override_user_agent]},
            timeout=httpx.Timeout(300.0),
        )

        # Create Anthropic client with custom base URL and HTTP client
        client = AsyncAnthropic(
            http_client=http_client,
            base_url=target_url,
            api_key=api_key,
        )

        # Build streaming parameters
        stream_kwargs: dict[str, Any] = {
            "model": target_req.get("model", "claude-sonnet-4-6"),
            "messages": target_req.get("messages", []),
            "max_tokens": target_req.get("max_tokens", 16384),
        }
        # Add optional parameters if present
        if target_req.get("system") is not None:
            stream_kwargs["system"] = target_req["system"]
        if target_req.get("tools") is not None:
            stream_kwargs["tools"] = target_req["tools"]
        if target_req.get("temperature") is not None:
            stream_kwargs["temperature"] = target_req["temperature"]

        # Start streaming from Claude
        print(f"[STREAM] Connecting to {target_url}...")
        async with client.messages.stream(**stream_kwargs) as stream:
            print("[STREAM] Connected, sending response headers...")

            # Send HTTP response headers for SSE
            status_line = "HTTP/1.1 200 OK\r\n"
            writer.write(status_line.encode())
            writer.write(b"Content-Type: text/event-stream\r\n")
            writer.write(b"Cache-Control: no-cache\r\n")
            writer.write(b"Connection: keep-alive\r\n")
            writer.write(b"Access-Control-Allow-Origin: *\r\n")
            writer.write(b"\r\n")
            await writer.drain()

            # Create stream state objects to track conversation state
            # The source adapter's create_stream_state() returns the appropriate
            # state class for the source format (e.g., GeminiStreamState)
            source_state = source_adapter.create_stream_state()
            target_state = source_adapter.create_stream_state()  # ClaudeStreamState

            # Process each streaming event from Claude
            event_count = 0
            async for event in stream:
                event_count += 1
                event_type = event.type

                # Log first few events for debugging
                if event_count <= 3:
                    print(f"[STREAM] Event #{event_count}: {event_type}")

                # Convert Claude event to dict format for the adapter
                event_dict: dict[str, Any] = {"type": event_type}

                # Attach additional event data if present
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

                # Parse Claude event to internal format
                internal_event = _parse_claude_event(event_dict, target_state)
                if internal_event is None:
                    continue  # Skip events that don't produce output

                # Convert to source format and send to client
                result = source_adapter.format_stream_event(
                    internal_event, source_state
                )
                if result:
                    try:
                        # SSE format: "data: <json>\r\n\r\n"
                        writer.write(f"data: {result}\r\n\r\n".encode())
                        await writer.drain()
                    except Exception as write_err:
                        print(f"[STREAM] Write error: {write_err}")
                        break

            # Send empty data to signal end of stream
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
    """
    Parse a Claude SDK event into the internal stream event format.

    Claude SDK sends different event types:
    - message_start: Initial message metadata
    - content_block_start: Start of a content block (text, tool_use)
    - content_block_delta: Incremental content (text delta, tool args delta)
    - message_delta: Usage data and stop reason

    This function converts these to our internal format so they can
    be re-formatted for the source client.

    Args:
        event: Event dict from Claude SDK
        state: ClaudeStreamState object for accumulating data

    Returns:
        InternalStreamEvent or None (if event doesn't produce output)
    """
    from providers.schema import InternalStreamEvent, ToolCall, Usage

    event_type = event.get("type", "")

    # message_start: Initialize message ID and model
    if event_type == "message_start":
        msg = event.get("message", {})
        state.message_id = msg.get("id", "")
        state.model = msg.get("model", "")
        return None  # No output to client yet

    # content_block_start: Initialize a new content block
    elif event_type == "content_block_start":
        block = event.get("content_block", {})
        block_type = block.get("type", "")
        index = event.get("index", 0)

        # Track tool use blocks for later
        if block_type == "tool_use":
            state.tool_call_accum[index] = {
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": "",
            }
        return None

    # content_block_delta: Incremental content updates
    elif event_type == "content_block_delta":
        delta = event.get("delta", {})
        delta_type = delta.get("type", "")
        index = event.get("index", 0)

        # Text delta: Return as text event
        if delta_type == "text_delta" and delta.get("text"):
            return InternalStreamEvent(type="text", text=delta["text"])

        # Thinking delta: Claude's extended thinking
        elif delta_type == "thinking_delta" and delta.get("thinking"):
            return InternalStreamEvent(type="thinking", thinking=delta["thinking"])

        # Input JSON delta: Accumulate tool arguments
        # Tool call arguments come in pieces, need to accumulate
        elif delta_type == "input_json_delta" and delta.get("partial_json"):
            if index in state.tool_call_accum:
                state.tool_call_accum[index]["arguments"] += delta["partial_json"]
            return None

    # message_delta: Final usage and stop reason
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
        # Tool calls complete before message_delta, so emit them here
        if state.tool_call_accum:
            if not hasattr(state, "pending_tool_events"):
                state.pending_tool_events = []
            for idx in sorted(state.tool_call_accum.keys()):
                accum = state.tool_call_accum[idx]
                args = {}
                try:
                    args = json.loads(accum["arguments"])
                except (json.JSONDecodeError, ValueError):
                    pass
                state.pending_tool_events.append(
                    InternalStreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=accum["id"], name=accum["name"], arguments=args
                        ),
                    )
                )
            state.tool_call_accum.clear()
            if state.pending_tool_events:
                return state.pending_tool_events.pop(0)

        # Drain any remaining pending tool events
        if hasattr(state, "pending_tool_events") and state.pending_tool_events:
            return state.pending_tool_events.pop(0)

        # Final event with usage and stop reason
        if stop_reason:
            return InternalStreamEvent(
                type="done",
                finish_reason=stop_reason,
                usage=state.usage,
            )

    return None


# =============================================================================
# GENERIC STREAMING HANDLER (for non-Claude providers)
# =============================================================================


async def _stream_generic(
    target_url: str,
    api_key: str,
    target_req: dict,
    target_adapter: Any,
    source_adapter: Any,
    writer: Any,
) -> None:
    """
    Handle streaming for non-Claude providers (OpenAI, Gemini, etc.).

    This is a generic SSE streaming handler that works with any provider
    that uses standard Server-Sent Events format:
    - data: {"delta": {"content": "..."}}\r\n\r\n
    - data: [DONE]\r\n\r\n

    The target_adapter parses events from the target provider.
    The source_adapter converts them to the format expected by the client.

    Args:
        target_url: Full URL to the streaming endpoint
        api_key: API key for authentication
        target_req: Formatted request dict
        target_adapter: Adapter to parse target provider's events
        source_adapter: Adapter to format events for source client
        writer: StreamWriter to send response to client
    """
    headers = target_adapter.get_headers(api_key or "")

    try:
        # Send request with streaming enabled
        async with aiohttp.ClientSession() as session:
            async with session.post(
                target_url,
                headers=headers,
                json=target_req,
            ) as resp:
                # Send response headers for SSE
                status_line = "HTTP/1.1 200 OK\r\n"
                writer.write(status_line.encode())
                writer.write(b"Content-Type: text/event-stream\r\n")
                writer.write(b"Cache-Control: no-cache\r\n")
                writer.write(b"Connection: keep-alive\r\n")
                writer.write(b"Access-Control-Allow-Origin: *\r\n")
                writer.write(b"\r\n")
                await writer.drain()

                # Create stream state objects for both adapters
                source_state = source_adapter.create_stream_state()
                target_state = target_adapter.create_stream_state()

                # Read SSE stream line by line
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()

                    # Skip non-data lines
                    if not line_str.startswith("data: "):
                        continue

                    # Extract data content (remove "data: " prefix)
                    data = line_str[6:]  # Remove "data: "
                    if data == "[DONE]":
                        break  # End of stream

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # Parse target event to internal format
                    internal_event = target_adapter.parse_stream_event(
                        event, target_state
                    )
                    if internal_event is None:
                        continue

                    # Convert to source format and send
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

                # Signal end of stream
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
