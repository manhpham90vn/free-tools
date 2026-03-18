"""
Transparent Passthrough Proxy for Antigravity MITM Proxy.

This module handles requests that should NOT be intercepted.
When the MITM proxy receives a request that doesn't match any
interception pattern (e.g., loading web UI, fetching user info),
it forwards the request transparently to the real server.

Key challenge: Since we've modified /etc/hosts to point the target
hostname to 127.0.0.1, we can't use normal DNS to reach the real server.
Instead, we use Google's public DNS (8.8.8.8) to resolve the real IP
address, bypassing /etc/hosts entirely.

Anti-loop protection: We add a custom header to forwarded requests
to detect if a request originated from our own proxy (which would
cause an infinite loop).
"""

# === Standard library imports ===
import ssl  # SSL context for outgoing HTTPS connections
from typing import Dict, Any  # Type hints

# === Third-party imports ===
import aiohttp  # Async HTTP client for forwarding requests
import dns.resolver  # DNS resolution using custom nameservers (bypasses /etc/hosts)

# === Internal module imports ===
from .utils import send_error_response  # Utility for sending error responses


# =============================================================================
# ANTI-LOOP DETECTION
# =============================================================================

# Custom header added to forwarded requests to detect proxy loops.
# If we receive a request with this header, it means the request
# came from our own proxy → drop it to prevent infinite loops.
LOOP_HEADER = "x-request-source"
LOOP_VALUE = "antigravity-mitm"


def resolve_real_ip(hostname: str) -> str:
    """
    Resolve the real IP address of a hostname using Google DNS (8.8.8.8).

    Since we've modified /etc/hosts to point intercepted hostnames to 127.0.0.1,
    normal DNS resolution would return 127.0.0.1 (our proxy).
    To reach the REAL server, we query Google's public DNS directly,
    which ignores /etc/hosts.

    Args:
        hostname: The hostname to resolve (e.g., "cloudcode-pa.googleapis.com")

    Returns:
        The real IP address as a string (e.g., "142.250.80.42")

    Raises:
        dns.resolver.NXDOMAIN: If the hostname doesn't exist
        dns.resolver.Timeout: If DNS query times out (5 second limit)
    """
    # Create a custom resolver that uses Google's public DNS servers
    resolver = dns.resolver.Resolver()
    resolver.nameservers = ["8.8.8.8", "8.8.4.4"]  # Google Public DNS
    resolver.lifetime = 5.0  # 5 second timeout for the entire query

    # Resolve the hostname to an A record (IPv4 address)
    answers = resolver.resolve(hostname, "A")
    # Return the first IP address from the response
    return str(answers[0])


def is_loop_request(headers: Dict[str, str]) -> bool:
    """
    Check if this request originated from our own proxy (anti-loop detection).

    When we forward a request to the real server, we add a custom header.
    If we receive a request with that header, it means the request
    somehow came back to us → it's a loop and should be dropped.

    Args:
        headers: HTTP headers from the incoming request

    Returns:
        True if this is a looped request (should be dropped)
    """
    return headers.get(LOOP_HEADER) == LOOP_VALUE


# =============================================================================
# PASSTHROUGH HANDLER
# =============================================================================


async def passthrough(
    method: str,
    path: str,
    headers: Dict[str, str],
    body: bytes,
    hostname: str,
    writer: Any,
) -> None:
    """
    Forward a request transparently to the real server.

    This function handles non-intercepted requests by:
    1. Checking for proxy loops (drop if detected)
    2. Resolving the real IP using Google DNS (bypasses /etc/hosts)
    3. Building a new request with cleaned headers
    4. Forwarding to the real server via HTTPS
    5. Streaming the response back to the original client

    Note: We disable SSL verification for the outgoing connection because
    we're connecting to the real server by IP address, not hostname.
    The hostname won't match the server's certificate.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: URL path from the request
        headers: HTTP headers from the request
        body: Request body as bytes
        hostname: Original target hostname (from SNI)
        writer: asyncio StreamWriter to send response back to client
    """
    # Anti-loop check: drop requests that came from our own proxy
    if is_loop_request(headers):
        print(f"[LOOP DETECTED] Request from self, dropping: {hostname}{path}")
        return

    # Resolve the real IP address using Google DNS
    try:
        real_ip = resolve_real_ip(hostname)
        print(f"[PASSTHROUGH] {hostname} -> {real_ip}{path}")
    except Exception as e:
        print(f"[ERROR] Failed to resolve {hostname}: {e}")
        await send_error_response(writer, 502, "Bad Gateway", "DNS resolution failed")
        return

    # Build the target URL using the real IP address
    target_url = f"https://{real_ip}{path}"

    # === Clean up headers for forwarding ===
    # Remove hop-by-hop headers (these are connection-specific and shouldn't be forwarded)
    forward_headers = {}
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    for key, value in headers.items():
        if key.lower() not in hop_by_hop:
            forward_headers[key] = value

    # Set the Host header to the original hostname (not the IP)
    # The real server needs this to identify which virtual host to serve
    forward_headers["host"] = hostname

    # Add our anti-loop header so we can detect if this request comes back to us
    forward_headers[LOOP_HEADER] = LOOP_VALUE

    # Create SSL context that doesn't verify certificates
    # We're connecting by IP address, so the hostname won't match the cert
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                target_url,
                headers=forward_headers,
                data=body if body else None,
                ssl=ssl_context,
                skip_auto_headers=["Host"],  # Don't override our Host header
                allow_redirects=False,  # Don't follow redirects (let client handle them)
            ) as resp:
                print(f"[PASSTHROUGH] Response: {resp.status} {resp.reason}")

                # Forward the HTTP status line
                status_line = f"HTTP/1.1 {resp.status} {resp.reason}\r\n"
                writer.write(status_line.encode())

                # Forward response headers
                # Skip hop-by-hop headers and content-encoding
                # (aiohttp auto-decompresses, so content-encoding would be wrong)
                for key, value in resp.headers.items():
                    if (
                        key.lower() not in hop_by_hop
                        and key.lower() != "content-encoding"
                    ):
                        writer.write(f"{key}: {value}\r\n".encode())

                # Close connection after response
                writer.write(b"Connection: close\r\n")
                writer.write(b"\r\n")
                await writer.drain()

                # Forward the response body
                resp_body = await resp.read()
                if resp_body:
                    writer.write(resp_body)
                    await writer.drain()

                # Debug logging for key endpoints
                # These endpoints are useful for debugging Antigravity integration
                if any(
                    ep in path
                    for ep in (
                        "loadCodeAssist",
                        "fetchUserInfo",
                        "fetchAvailableModels",
                    )
                ):
                    try:
                        import json

                        body_str = resp_body.decode(errors="replace")
                        parsed = json.loads(body_str)
                        print(
                            f"[PASSTHROUGH] Body: {json.dumps(parsed, indent=2)[:10]}"
                        )
                    except Exception:
                        print(
                            f"[PASSTHROUGH] Body (raw): {resp_body[:10].decode(errors='replace')}"
                        )

                print(f"[PASSTHROUGH] Forwarded {len(resp_body)} bytes")

    except aiohttp.ClientError as e:
        print(f"[ERROR] Passthrough failed: {e}")
        await send_error_response(writer, 502, "Bad Gateway", str(e))
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
