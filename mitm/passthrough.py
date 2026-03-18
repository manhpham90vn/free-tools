"""
Transparent passthrough proxy for Antigravity MITM proxy.
Forwards non-intercepted requests to the real server, bypassing /etc/hosts.
"""

import ssl
from typing import Dict, Any

import aiohttp
import dns.resolver


# Custom header to detect loops
LOOP_HEADER = "x-request-source"
LOOP_VALUE = "antigravity-mitm"


def resolve_real_ip(hostname: str) -> str:
    """
    Resolve the real IP of a hostname using Google DNS (8.8.8.8),
    bypassing /etc/hosts.

    Args:
        hostname: The hostname to resolve

    Returns:
        The resolved IP address
    """
    resolver = dns.resolver.Resolver()
    resolver.nameservers = ["8.8.8.8", "8.8.4.4"]
    answers = resolver.resolve(hostname, "A")
    return str(answers[0])


def is_loop_request(headers: Dict[str, str]) -> bool:
    """Check if this request originated from our own proxy (anti-loop)."""
    return headers.get(LOOP_HEADER) == LOOP_VALUE


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

    Args:
        method: HTTP method
        path: Request path
        headers: Request headers
        body: Request body
        hostname: Original target hostname
        writer: asyncio StreamWriter to write response to
    """
    if is_loop_request(headers):
        print(f"[LOOP DETECTED] Request from self, dropping: {hostname}{path}")
        return

    try:
        # Resolve real IP using DNS that bypasses /etc/hosts
        real_ip = resolve_real_ip(hostname)
        print(f"[PASSTHROUGH] {hostname} -> {real_ip}{path}")
    except Exception as e:
        print(f"[ERROR] Failed to resolve {hostname}: {e}")
        await send_error_response(writer, 502, "Bad Gateway", "DNS resolution failed")
        return

    # Build request to real server
    target_url = f"https://{real_ip}{path}"

    # Forward headers (remove hop-by-hop and add loop header)
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

    forward_headers[LOOP_HEADER] = LOOP_VALUE

    # Create SSL context that doesn't verify (we're MITM ourselves)
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
                skip_auto_headers=["Host"],
                allow_redirects=False,
            ) as resp:
                print(f"[PASSTHROUGH] Response: {resp.status} {resp.reason}")

                # Forward status line
                status_line = f"HTTP/1.1 {resp.status} {resp.reason}\r\n"
                writer.write(status_line.encode())

                # Forward headers
                for key, value in resp.headers.items():
                    if key.lower() not in hop_by_hop:
                        writer.write(f"{key}: {value}\r\n".encode())

                writer.write(b"Connection: close\r\n")
                writer.write(b"\r\n")
                await writer.drain()

                # Forward body
                resp_body = await resp.read()
                if resp_body:
                    writer.write(resp_body)
                    await writer.drain()

                # Debug: log response body for key endpoints
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
                            f"[PASSTHROUGH] Body: {json.dumps(parsed, indent=2)[:5000]}"
                        )
                    except Exception:
                        print(
                            f"[PASSTHROUGH] Body (raw): {resp_body[:1000].decode(errors='replace')}"
                        )

                print(f"[PASSTHROUGH] Forwarded {len(resp_body)} bytes")

    except aiohttp.ClientError as e:
        print(f"[ERROR] Passthrough failed: {e}")
        await send_error_response(writer, 502, "Bad Gateway", str(e))
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
