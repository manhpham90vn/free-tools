"""
MITM HTTPS Server for Antigravity Proxy.

This module implements the core TLS interception server that:
1. Listens on port 443 (HTTPS)
2. Uses SNI (Server Name Indication) to detect which domain the client is connecting to
3. Generates on-the-fly certificates signed by our Root CA for that domain
4. Handles the TLS handshake with the client using the generated certificate
5. Parses the HTTP request and decides whether to intercept or passthrough

Key components:
- MITMServer class: Main server implementation using asyncio
- get_ssl_context(): Creates SSL contexts with dynamically generated certificates
- _sni_callback(): SSL callback that switches certificate based on SNI hostname
- handle_client(): Processes each incoming client connection
"""

# === Standard library imports ===
import asyncio  # Async I/O for handling multiple concurrent connections
import ssl  # SSL/TLS context creation and configuration
import tempfile  # Temporary files for passing certs to the ssl module
import os  # File operations (reading/writing temp files, cleanup)
from typing import Dict, Any  # Type hints for configuration dictionary

# === Internal module imports ===
import cert  # Certificate generation (Root CA, leaf certificates)
from logger import get_logger  # Structured logging
from .handler import (
    should_intercept,  # Determines if a request should be intercepted
    forward_to_target,  # Forwards non-streaming requests to target
    forward_to_target_streaming,  # Forwards streaming requests to target
)
from .passthrough import passthrough  # Forwards non-intercepted requests to real server

# Module-level logger
log = get_logger("mitm.server")


class MITMServer:
    """
    MITM (Man-in-the-Middle) HTTPS Server.

    This server intercepts HTTPS traffic by:
    1. Presenting a dynamically-generated certificate signed by our Root CA
    2. Terminating the TLS connection from the client
    3. Parsing the HTTP request inside the encrypted tunnel
    4. Forwarding the request to the actual target (or proxying transparently)

    The server uses SNI (Server Name Indication) from the TLS handshake
    to determine which domain the client intended to connect to, then generates
    a certificate specifically for that domain.

    Attributes:
        config: Configuration dictionary containing target_provider, model_mapping, etc.
        cert_dir: Directory where Root CA certificates are stored
        ca_key: Root CA private key (used to sign leaf certificates)
        ca_cert: Root CA certificate (presented to clients)
        _ssl_contexts: Cache of SSL contexts per hostname (performance optimization)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MITM server with configuration.

        Args:
            config: Application configuration dictionary. Should contain:
                - cert_dir: Path to store/load Root CA files
                - target_provider: Which LLM provider to use (claude, openai, etc.)
                - model_mapping: Dictionary mapping source models to target models
                - hosts: List of hostnames to intercept
        """
        self.config = config
        # cert_dir must be pre-resolved to an absolute path via _get_cert_dir()
        self.cert_dir = config["cert_dir"]

        # Load or generate the Root CA keypair
        # If Root CA files exist in cert_dir, load them
        # Otherwise, generate new ones
        self.ca_key, self.ca_cert = cert.load_or_create_root_ca(self.cert_dir)

        # SSL context cache: maps hostname -> SSL context with that hostname's certificate
        # Caching avoids regenerating certificates for the same hostname
        self._ssl_contexts: Dict[str, ssl.SSLContext] = {}

    def get_ssl_context(self, hostname: str) -> ssl.SSLContext:
        """
        Get or create an SSL context for the given hostname.

        This is the core of the MITM certificate system:
        1. Check if we already have a cached SSL context for this hostname
        2. If not, generate a new leaf certificate for this hostname
        3. Create an SSL context and load the certificate and key
        4. Cache it for future requests to the same hostname

        The ssl module requires file paths to load certificates, so we use
        tempfile.NamedTemporaryFile to create the cert/key files.

        Args:
            hostname: The SNI hostname (e.g., "cloudcode-pa.googleapis.com")

        Returns:
            Configured SSL context for the hostname, ready to use in TLS handshake
        """
        # Check cache first to avoid regenerating certificates
        if hostname not in self._ssl_contexts:
            # Generate a new leaf certificate for this specific domain
            # Signed by our Root CA, so clients will trust it if they trust our Root CA
            cert_pem, key_pem = cert.generate_leaf_cert(
                hostname, self.ca_key, self.ca_cert
            )

            # Create SSL context for SERVER mode (we're the server in the TLS connection)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

            # Set strong ciphers to ensure compatibility and security
            # ECDHE = Elliptic Curve Diffie-Hellman (forward secrecy)
            # AESGCM and CHACHA20 = Authenticated encryption
            ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")

            # === Workaround: ssl module requires file paths, not raw bytes ===
            # We need to write cert/key to temp files because
            # ssl.SSLContext.load_cert_chain() only accepts file paths
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
                # Load the certificate and private key into the SSL context
                ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
            finally:
                # Clean up temp files immediately after loading
                # The SSL context holds the cert/key in memory
                try:
                    os.unlink(cert_path)
                    os.unlink(key_path)
                except Exception:
                    pass  # Ignore cleanup errors

            # Register SNI callback - this is called during TLS handshake
            # when the client sends the hostname it's trying to connect to
            ctx.sni_callback = self._sni_callback

            # Cache for future requests to the same hostname
            self._ssl_contexts[hostname] = ctx

        return self._ssl_contexts[hostname]

    def _sni_callback(self, ssl_object, server_name, ssl_context):
        """
        SNI (Server Name Indication) callback for dynamic certificate selection.

        During TLS handshake, the client sends the hostname it wants to connect to
        (the Server Name Indication extension). This callback is invoked with that
        name, allowing us to switch to the correct certificate for that domain.

        How it works:
        1. Client initiates TLS handshake and sends SNI (e.g., "cloudcode-pa.googleapis.com")
        2. Python's ssl module calls this callback with the server_name
        3. We generate (or get from cache) an SSL context with the correct certificate
        4. We replace ssl_object.context with our new context

        This enables the MITM server to serve multiple domains from a single listener.

        Args:
            ssl_object: The SSL object being created for this connection
            server_name: The hostname the client is trying to reach (may be bytes or str)
            ssl_context: The original SSL context (not used, we replace it)
        """
        try:
            if server_name:
                # Handle both bytes (older Python versions) and str formats
                hostname = (
                    server_name
                    if isinstance(server_name, str)
                    else server_name.decode()
                )
                # Replace the SSL context with one containing the correct certificate
                ssl_object.context = self.get_ssl_context(hostname)
        except Exception as e:
            log.error("SNI callback error: {e}", e=e)

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Handle an incoming client connection.

        This is the main request handler. It:
        1. Extracts the hostname from TLS SNI
        2. Reads and parses the HTTP request
        3. Decides whether to intercept (process with LLM) or passthrough (forward transparently)
        4. Sends the response back to the client

        Args:
            reader: Async stream reader for reading client data
            writer: Async stream writer for sending response to client
        """
        # Get client address for logging
        peername = writer.get_extra_info("peername")
        log.debug("New connection from {peername}", peername=peername)

        try:
            # === Step 1: Extract hostname from TLS SNI ===
            ssl_object = writer.get_extra_info("ssl_object")
            hostname = None

            if ssl_object:
                try:
                    # server_hostname is the SNI hostname sent by the client
                    sni = ssl_object.server_hostname
                    if sni:
                        # Convert bytes to str if necessary
                        hostname = sni if isinstance(sni, str) else sni.decode()
                        log.sni("hostname={hostname}", hostname=hostname)
                except Exception as e:
                    log.sni("Error getting hostname: {e}", e=e)

            # Fallback: if no SNI, use the first configured host as default
            if not hostname:
                hostname = self.config.get("hosts", ["unknown"])[0]
                log.sni("No hostname, fallback to {hostname}", hostname=hostname)

            # === Step 2: Read HTTP request ===
            data = await reader.read(65536)  # Read up to 64KB
            if not data:
                # Client sent empty request, close connection
                log.debug("Empty read, closing")
                writer.close()
                return

            log.req("{n} bytes received", n=len(data))

            # === Step 3: Parse request line ===
            try:
                # HTTP request line format: "METHOD /path HTTP/1.1"
                # Split by \r\n first line is the request line
                request_line = data.decode(errors="replace").split("\r\n")[0]
                method, path, _ = request_line.split(" ")
                log.req(
                    "{method} {hostname}{path}",
                    method=method,
                    hostname=hostname,
                    path=path,
                )
            except (ValueError, UnicodeDecodeError) as e:
                # Failed to parse request line - malformed request
                log.error("Failed to parse request line: {e}", e=e)
                log.debug(
                    "Raw data (first 200): {data}",
                    data=data[:200].decode(errors="replace"),
                )
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            # === Step 4: Parse HTTP headers ===
            try:
                headers = {}
                # Find the boundary between headers and body (empty line: \r\n\r\n)
                body_start = data.find(b"\r\n\r\n")
                if body_start != -1:
                    # Extract header section (everything before the empty line)
                    header_section = data[:body_start].decode(errors="replace")
                    # Body starts after the empty line
                    body = data[body_start + 4 :]

                    # Check for chunked transfer encoding (common with SSE/streaming)
                    # and Content-Length (for requests with body)
                    is_chunked = False
                    content_length = 0

                    # Parse each header line
                    for line in header_section.split("\r\n")[1:]:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            headers[key.strip().lower()] = value.strip()

                            # Detect transfer-encoding: chunked
                            if (
                                key.strip().lower() == "transfer-encoding"
                                and value.strip().lower() == "chunked"
                            ):
                                is_chunked = True

                            # Extract Content-Length
                            if key.strip().lower() == "content-length":
                                content_length = int(value.strip())

                    # === Handle chunked transfer encoding ===
                    if is_chunked:
                        # HTTP chunked encoding: each chunk has size + data + \r\n
                        # Format: <size in hex>\r\n<data>\r\n ... 0\r\n\r\n
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
                                # Find the \r\n that separates size from data
                                crlf = buf.find(b"\r\n")
                                size_str = buf[:crlf].decode(errors="replace").strip()

                                if not size_str:
                                    # Empty line, skip it
                                    buf = buf[crlf + 2 :]
                                    continue

                                try:
                                    chunk_size = int(size_str, 16)  # Parse hex size
                                except ValueError:
                                    break  # Invalid size, stop parsing

                                if chunk_size == 0:
                                    break  # Last chunk (size 0), we're done

                                # Calculate how many bytes we need (data + trailing \r\n)
                                needed = crlf + 2 + chunk_size + 2
                                # Keep reading until we have the full chunk
                                while len(buf) < needed:
                                    more = await reader.read(65536)
                                    if not more:
                                        break
                                    buf += more

                                # Extract chunk data (skip the size line and trailing \r\n)
                                data_start = crlf + 2
                                decoded.extend(
                                    buf[data_start : data_start + chunk_size]
                                )
                                # Remove processed chunk from buffer
                                buf = buf[data_start + chunk_size + 2 :]
                                continue
                            break  # No more \r\n in buffer, EOF

                        body = bytes(decoded)
                        log.req("Reassembled chunked body: {n} bytes", n=len(body))

                    # === Handle Content-Length (regular body) ===
                    elif content_length > 0:
                        # Read remaining body bytes to match Content-Length
                        remaining = content_length - len(body)
                        if remaining > 0:
                            log.req("Reading remaining {n} body bytes...", n=remaining)
                        while remaining > 0:
                            chunk = await reader.read(min(remaining, 65536))
                            if not chunk:
                                break
                            body += chunk
                            remaining -= len(chunk)

                    # Log parsed headers and body size
                    log.req(
                        "Headers: {headers}...", headers=dict(list(headers.items())[:5])
                    )
                    log.req("Body size: {n} bytes", n=len(body))
                else:
                    # No body (no \r\n\r\n found)
                    body = b""

            except Exception as e:
                import traceback

                log.error("Failed to parse headers: {e}", e=e)
                traceback.print_exc()
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            # === Step 5: Decide to intercept or passthrough ===
            # Check if this request should be intercepted (LLM API call)
            if should_intercept(path):
                log.intercept(
                    "{method} {hostname}{path}",
                    method=method,
                    hostname=hostname,
                    path=path,
                )

                # Handle streaming vs non-streaming requests differently
                # Streaming uses SSE (Server-Sent Events) for real-time response
                if ":streamGenerateContent" in path:
                    # Streaming: forward and stream response back to client
                    await forward_to_target_streaming(
                        method, path, headers, body, self.config, writer
                    )
                else:
                    # Non-streaming: forward and get complete response
                    status, resp_headers, resp_body = await forward_to_target(
                        method, path, headers, body, self.config
                    )

                    log.resp(
                        "Status: {status}, Body: {n} bytes",
                        status=status,
                        n=len(resp_body),
                    )

                    # === Send HTTP response to client ===
                    status_line = f"HTTP/1.1 {status} OK\r\n"
                    writer.write(status_line.encode())

                    # Forward response headers (except hop-by-hop headers)
                    for key, value in resp_headers.items():
                        if key.lower() not in ("transfer-encoding", "connection"):
                            writer.write(f"{key}: {value}\r\n".encode())

                    # Set Content-Length and close connection
                    writer.write(f"Content-Length: {len(resp_body)}\r\n".encode())
                    writer.write(b"Connection: close\r\n")
                    writer.write(b"\r\n")
                    writer.write(resp_body)
                    await writer.drain()
                    writer.close()
            else:
                # Not an intercepted endpoint - passthrough to real server
                log.passthrough(
                    "{method} {hostname}{path}",
                    method=method,
                    hostname=hostname,
                    path=path,
                )
                await passthrough(method, path, headers, body, hostname, writer)

        except Exception as e:
            import traceback

            log.error("Handler error: {e}", e=e)
            traceback.print_exc()
            try:
                writer.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                await writer.drain()
            except Exception:
                pass
            writer.close()

    async def start(self, host: str = "0.0.0.0", port: int = 443) -> None:
        """
        Start the MITM server and begin accepting connections.

        This creates an asyncio TCP server with SSL enabled. Each incoming
        connection triggers handle_client() in a new coroutine.

        The server needs to present an SSL certificate immediately upon startup,
        so we generate a default certificate for the first configured hostname.
        The actual per-hostname certificate switching happens in the SNI callback.

        Args:
            host: IP address to bind to (0.0.0.0 = all interfaces)
            port: Port to listen on (443 = standard HTTPS, requires root)
        """
        # Get a default hostname for initial certificate generation
        # This certificate is used until SNI tells us the actual hostname
        default_host = self.config.get("hosts", ["daily-cloudcode-pa.googleapis.com"])[
            0
        ]
        # Generate SSL context with default certificate
        default_ssl = self.get_ssl_context(default_host)

        try:
            # Create async server with SSL enabled
            server = await asyncio.start_server(
                self.handle_client,  # Callback for each incoming connection
                host,
                port,
                reuse_address=True,  # Allow reusing the address (TIME_WAIT)
                ssl=default_ssl,  # SSL context for TLS
            )
        except OSError as e:
            if e.errno == 98:  # Address already in use (EADDRINUSE)
                log.error("Port {port} is already in use.", port=port)
                log.error("Please kill the existing process first:")
                log.error("  sudo fuser -k {port}/tcp", port=port)
                raise SystemExit(1)
            raise  # Re-raise other OS errors

        log.banner(
            "MITM Server listening on https://{host}:{port}", host=host, port=port
        )
        log.banner(
            "Intercepting: {hosts}", hosts=", ".join(self.config.get("hosts", []))
        )
        log.banner("Press Ctrl+C to stop")

        # Keep the server running forever
        # The async with block ensures proper cleanup on exit
        async with server:
            await server.serve_forever()
