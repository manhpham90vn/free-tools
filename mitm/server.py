"""
MITM HTTPS server for Antigravity proxy.
Listens on port 443 and intercepts Gemini API requests.
"""

import asyncio
import ssl
import tempfile
import os
from typing import Dict, Any

from . import cert
from .handler import should_intercept, forward_to_target, forward_to_target_streaming
from .passthrough import passthrough


class MITMServer:
    """MITM HTTPS Server that intercepts Gemini API requests."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MITM server.

        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.cert_dir = config.get("cert_dir", "~/.free-antigravity")

        # Load or create Root CA
        self.ca_key, self.ca_cert = cert.load_or_create_root_ca(self.cert_dir)

        # SSL context for server (will be configured per-connection via SNI)
        self._ssl_contexts: Dict[str, ssl.SSLContext] = {}

    def get_ssl_context(self, hostname: str) -> ssl.SSLContext:
        """
        Get or create an SSL context for the given hostname.

        Args:
            hostname: The SNI hostname

        Returns:
            Configured SSL context for the hostname
        """
        if hostname not in self._ssl_contexts:
            # Generate leaf certificate
            cert_pem, key_pem = cert.generate_leaf_cert(
                hostname, self.ca_key, self.ca_cert
            )

            # Create SSL context
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")

            # Load certificate and key from temp files (ssl module requires file paths)
            # We need to write to files because ssl.load_cert_chain needs paths
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pem", delete=False
            ) as cert_file:
                cert_file.write(cert_pem)
                cert_path = cert_file.name

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pem", delete=False
            ) as key_file:
                key_file.write(key_pem)
                key_path = key_file.name

            try:
                ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
            finally:
                # Clean up temp files
                try:
                    os.unlink(cert_path)
                    os.unlink(key_path)
                except Exception:
                    pass

            # Set SNI callback
            ctx.sni_callback = self._sni_callback

            self._ssl_contexts[hostname] = ctx

        return self._ssl_contexts[hostname]

    def _sni_callback(self, ssl_object, server_name, ssl_context):
        """SNI callback: (ssl_object, server_name, original_context)."""
        try:
            if server_name:
                hostname = (
                    server_name
                    if isinstance(server_name, str)
                    else server_name.decode()
                )
                ssl_object.context = self.get_ssl_context(hostname)
        except Exception as e:
            print(f"[ERROR] SNI callback error: {e}")

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming client connection."""
        peername = writer.get_extra_info("peername")
        print(f"\n[CONN] New connection from {peername}")

        try:
            # Get the hostname from TLS SNI via the ssl object
            ssl_object = writer.get_extra_info("ssl_object")
            hostname = None

            if ssl_object:
                try:
                    sni = ssl_object.server_hostname
                    if sni:
                        hostname = sni if isinstance(sni, str) else sni.decode()
                        print(f"[SNI] hostname={hostname}")
                except Exception as e:
                    print(f"[SNI] Error getting hostname: {e}")

            if not hostname:
                # Fallback: use first configured host
                hostname = self.config.get("hosts", ["unknown"])[0]
                print(f"[SNI] No hostname, fallback to {hostname}")

            # Read HTTP request
            data = await reader.read(65536)
            if not data:
                print("[CONN] Empty read, closing")
                writer.close()
                return

            print(f"[RECV] {len(data)} bytes")

            # Parse request line
            try:
                request_line = data.decode(errors="replace").split("\r\n")[0]
                method, path, _ = request_line.split(" ")
                print(f"[REQ] {method} {hostname}{path}")
            except (ValueError, UnicodeDecodeError) as e:
                print(f"[ERROR] Failed to parse request line: {e}")
                print(
                    f"[ERROR] Raw data (first 200): {data[:200].decode(errors='replace')}"
                )
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            # Parse headers
            try:
                headers = {}
                body_start = data.find(b"\r\n\r\n")
                if body_start != -1:
                    header_section = data[:body_start].decode(errors="replace")
                    body = data[body_start + 4 :]

                    # Check for chunked transfer encoding
                    is_chunked = False
                    content_length = 0
                    for line in header_section.split("\r\n")[1:]:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            headers[key.strip().lower()] = value.strip()
                            if (
                                key.strip().lower() == "transfer-encoding"
                                and value.strip().lower() == "chunked"
                            ):
                                is_chunked = True
                            if key.strip().lower() == "content-length":
                                content_length = int(value.strip())

                    # Handle chunked transfer encoding
                    if is_chunked:
                        # Parse chunks incrementally - stop at terminator (0\r\n)
                        decoded = bytearray()
                        buf = body

                        while True:
                            # Ensure we have a chunk size line
                            while b"\r\n" not in buf:
                                more = await reader.read(65536)
                                if not more:
                                    break
                                buf += more
                            else:
                                crlf = buf.find(b"\r\n")
                                size_str = buf[:crlf].decode(errors="replace").strip()
                                if not size_str:
                                    buf = buf[crlf + 2 :]
                                    continue

                                try:
                                    chunk_size = int(size_str, 16)
                                except ValueError:
                                    break

                                if chunk_size == 0:
                                    break  # Done

                                # Read until we have full chunk + trailing \r\n
                                needed = crlf + 2 + chunk_size + 2
                                while len(buf) < needed:
                                    more = await reader.read(65536)
                                    if not more:
                                        break
                                    buf += more

                                # Extract chunk data
                                data_start = crlf + 2
                                decoded.extend(
                                    buf[data_start : data_start + chunk_size]
                                )
                                buf = buf[data_start + chunk_size + 2 :]
                                continue
                            break  # EOF reached

                        body = bytes(decoded)
                        print(f"[CHUNKED] Reassembled body: {len(body)} bytes")

                    # Read remaining body bytes if Content-Length is specified
                    elif content_length > 0:
                        remaining = content_length - len(body)
                        if remaining > 0:
                            print(f"[RECV] Reading remaining {remaining} body bytes...")
                        while remaining > 0:
                            chunk = await reader.read(min(remaining, 65536))
                            if not chunk:
                                break
                            body += chunk
                            remaining -= len(chunk)

                    print(f"[REQ] Headers: {dict(list(headers.items())[:5])}...")
                    print(f"[REQ] Body size: {len(body)} bytes")
                else:
                    body = b""
            except Exception as e:
                import traceback

                print(f"[ERROR] Failed to parse headers: {e}")
                traceback.print_exc()
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            # Check if we should intercept this request
            if should_intercept(path):
                print(f"[INTERCEPT] {method} {hostname}{path}")

                # Handle streaming vs non-streaming
                if ":streamGenerateContent" in path:
                    await forward_to_target_streaming(
                        method, path, headers, body, self.config, writer
                    )
                else:
                    status, resp_headers, resp_body = await forward_to_target(
                        method, path, headers, body, self.config
                    )

                    print(f"[RESP] Status: {status}, Body: {len(resp_body)} bytes")

                    # Send response
                    status_line = f"HTTP/1.1 {status} OK\r\n"
                    writer.write(status_line.encode())
                    for key, value in resp_headers.items():
                        if key.lower() not in ("transfer-encoding", "connection"):
                            writer.write(f"{key}: {value}\r\n".encode())
                    writer.write(f"Content-Length: {len(resp_body)}\r\n".encode())
                    writer.write(b"Connection: close\r\n")
                    writer.write(b"\r\n")
                    writer.write(resp_body)
                    await writer.drain()
                    writer.close()
            else:
                print(f"[PASSTHROUGH] {method} {hostname}{path}")
                await passthrough(method, path, headers, body, hostname, writer)

        except Exception as e:
            import traceback

            print(f"[ERROR] Handler error: {e}")
            traceback.print_exc()
            try:
                writer.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                await writer.drain()
            except Exception:
                pass
            writer.close()

    async def start(self, host: str = "0.0.0.0", port: int = 443) -> None:
        """
        Start the MITM server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        # Create a default SSL context for server startup
        # We'll regenerate certs per hostname after the TLS handshake
        default_host = self.config.get("hosts", ["daily-cloudcode-pa.googleapis.com"])[
            0
        ]
        default_ssl = self.get_ssl_context(default_host)

        try:
            server = await asyncio.start_server(
                self.handle_client,
                host,
                port,
                reuse_address=True,
                ssl=default_ssl,
            )
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"Error: Port {port} is already in use.")
                print("Please kill the existing process first:")
                print(f"  sudo fuser -k {port}/tcp")
                raise SystemExit(1)
            raise

        print(f"[*] MITM Server listening on https://{host}:{port}")
        print(f"[*] Intercepting: {', '.join(self.config.get('hosts', []))}")
        print("[*] Press Ctrl+C to stop")

        async with server:
            await server.serve_forever()
