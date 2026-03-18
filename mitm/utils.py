"""
Shared Utilities for MITM Proxy Modules.

This module contains common utility functions used across the MITM proxy
components, primarily for error handling and response formatting.
"""

# === Standard library imports ===
from html import escape  # HTML entity escaping to prevent XSS attacks
from typing import Any  # Type hints for generic types


async def send_error_response(
    writer: Any,
    status_code: int,
    reason: str,
    message: str,
) -> None:
    """
    Send an HTML error response to the HTTP client.

    This is used to return error pages when something goes wrong
    during request processing (e.g., invalid JSON, target unreachable).

    IMPORTANT: This function is XSS-safe. Both the reason and message
    are HTML-escaped before being inserted into the response body.
    This prevents Cross-Site Scripting (XSS) attacks if the error
    message contains user-supplied data.

    Args:
        writer: asyncio StreamWriter to send the response
        status_code: HTTP status code (e.g., 400, 500)
        reason: HTTP reason phrase (e.g., "Bad Request", "Internal Server Error")
        message: Error message to display in the body

    Example response:
        HTTP/1.1 400 Bad Request
        Content-Type: text/html
        Content-Length: 102

        <!DOCTYPE html>
        <html><body>
        <h1>400 Bad Request</h1>
        <p>Invalid JSON: ...</p>
        </body></html>
    """
    # Escape HTML entities in user-controlled strings to prevent XSS
    # & → &amp;  < → &lt;  > → &gt;  " → &quot;  ' → &#x27;
    safe_reason = escape(str(reason))
    safe_message = escape(str(message))

    # Build HTML response body
    body = f"""<!DOCTYPE html>
<html><body>
<h1>{status_code} {safe_reason}</h1>
<p>{safe_message}</p>
</body></html>"""

    # Build and send HTTP response
    status_line = f"HTTP/1.1 {status_code} {reason}\r\n"
    writer.write(status_line.encode())
    writer.write(b"Content-Type: text/html\r\n")
    writer.write(f"Content-Length: {len(body)}\r\n".encode())
    writer.write(b"Connection: close\r\n")
    writer.write(b"\r\n")
    writer.write(body.encode())
    await writer.drain()
    writer.close()
