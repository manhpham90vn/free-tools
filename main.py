#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free Tools - MITM Proxy.

This program acts as a Man-in-the-Middle proxy that intercepts
Gemini API requests and forwards them to a configurable custom endpoint.

Purpose:
- Allow using Antigravity, Cursor, Claude Code, Codex, Gemini CLI with any LLM provider
- Support multiple token sources: Claude, Gemini, OpenAI, Ollama
- Translate request/response formats between different providers automatically
"""

# === Standard library imports ===
import argparse  # Command-line argument parsing (subcommands: start, stop, status, etc.)
import asyncio  # Async I/O framework for running the MITM server
import os  # OS-level operations (env vars, process management, file paths)
import sys  # System-level functions (exit codes, argv, executable path)
from pathlib import Path  # Cross-platform file path handling

# === Third-party imports ===
import yaml  # YAML parser for reading config.yaml
from dotenv import load_dotenv  # Load environment variables from .env file

# === Internal module imports ===
# hostsutil: Manages DNS spoofing by modifying /etc/hosts
# Redirects target hostnames (e.g., cloudcode-pa.googleapis.com) to 127.0.0.1
from hostsutil import add_hosts, remove_hosts, flush_dns_cache, is_enabled

# cert: SSL certificate management for TLS interception
# Generates a Root CA and per-domain leaf certificates so the proxy can
# decrypt HTTPS traffic from the client
from cert import (
    install_ca,  # Generate Root CA (if needed) and install into system trust store
    uninstall_ca,  # Remove Root CA from trust store and optionally delete files
    trust_ca,  # Add Root CA to system trust store (requires root)
    untrust_ca,  # Remove Root CA from system trust store
    ca_exists,  # Check if Root CA certificate files exist on disk
    is_trusted,  # Check if Root CA is currently trusted by the system
    get_ca_cert_path,  # Get the filesystem path to the Root CA .crt file
)

# mitm/server: The core MITM HTTPS server that handles TLS interception,
# request parsing, provider detection, and response streaming
from mitm.server import MITMServer


# =============================================================================
# INITIALIZATION
# =============================================================================

# Determine the directory containing this script (resolved to absolute path).
# Used as the base for locating .env and config.yaml files.
_script_dir = Path(__file__).resolve().parent

# When running under sudo, env vars may be stripped for security.
# _FA_DOTENV is set before sudo to preserve the original .env path.
# Falls back to .env in the script directory if not set.
_dotenv_path = os.environ.get("_FA_DOTENV", str(_script_dir / ".env"))

# Load environment variables from .env file into os.environ.
# This allows configuring API keys and endpoints without modifying config.yaml.
# Example .env contents:
#   ANTHROPIC_BASE_URL=https://api.anthropic.com/v1/messages
#   ANTHROPIC_AUTH_TOKEN=sk-ant-xxx
load_dotenv(_dotenv_path)


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load and parse the YAML configuration file.

    The config file controls:
    - target_provider: Which LLM backend to use (claude, openai, gemini, ollama)
    - hosts: Which hostnames to intercept via DNS spoofing
    - model_mapping: How to map source model names to target model names
    - listen_port: Which port the MITM server listens on (default: 443)
    - cert_dir: Where to store Root CA and leaf certificates

    Args:
        config_path: Path to the YAML config file (default: config.yaml)

    Returns:
        Dictionary containing all configuration values

    Raises:
        SystemExit: If the config file does not exist
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # yaml.safe_load prevents arbitrary code execution from YAML files
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# ROOT PRIVILEGE MANAGEMENT
# =============================================================================


def ensure_root():
    """
    Ensure the script is running with root privileges. If not, re-execute with sudo.

    Root is required for:
    - Modifying /etc/hosts (DNS spoofing to redirect traffic to localhost)
    - Installing Root CA into the system trust store
    - Binding to port 443 (privileged port, requires root on Linux)

    How it works:
    1. Check if already root (euid == 0) → return immediately
    2. Check if we're in a failed sudo loop (_FA_SUDO set) → exit with error
    3. Resolve all paths to absolute (sudo doesn't preserve cwd)
    4. Pass critical env vars through sudo using `env` command
    5. Replace current process with sudo-elevated version via os.execvp
    """
    # os.geteuid() returns the effective user ID; 0 means root
    if os.geteuid() == 0:
        return  # Already running as root, nothing to do

    # _FA_SUDO is set by us before calling sudo. If it's already set,
    # it means sudo was called but we still don't have root → something is wrong
    if os.environ.get("_FA_SUDO"):
        print("Error: Failed to obtain root privileges.")
        sys.exit(1)

    print("[*] Requesting root privileges via sudo...")

    # === Pre-sudo path resolution ===
    # sudo may change HOME to /root and strip most env vars for security.
    # We must resolve all user-relative paths (~/) BEFORE calling sudo.

    # Resolve ~/.free-tools to absolute path (e.g., /home/user/.free-tools)
    cert_dir = str(Path("~/.free-tools").expanduser().resolve())
    # Resolve .env file path to absolute
    dotenv_path = str((_script_dir / ".env").resolve())

    # sudo doesn't preserve the current working directory,
    # so we need absolute paths for the script and config file
    script = str(Path(__file__).resolve())
    cwd = os.getcwd()

    # === Rebuild command-line arguments with absolute paths ===
    # If user passed `-c some/relative/config.yaml`, resolve it to absolute
    args = sys.argv[1:]
    resolved_args = []
    i = 0
    config_found = False

    while i < len(args):
        if args[i] in ("-c", "--config") and i + 1 < len(args):
            # Found config flag → resolve the config path to absolute
            resolved_args.append(args[i])
            resolved_args.append(str(Path(args[i + 1]).resolve()))
            config_found = True
            i += 2  # Skip both the flag and its value
        else:
            resolved_args.append(args[i])
            i += 1

    # If no --config was specified, inject the default config path (absolute)
    if not config_found:
        resolved_args = [
            "--config",
            str(Path(cwd, "config.yaml").resolve()),
        ] + resolved_args

    # === Build environment variables to pass through sudo ===
    # sudo strips most env vars for security. We use `env` command to set them.
    # Final command: sudo env VAR1=val1 VAR2=val2 python3 /abs/path/main.py ...
    env_args = [
        "_FA_SUDO=1",  # Flag to detect sudo loop
        f"_FA_CERT_DIR={cert_dir}",  # Pre-resolved cert directory
        f"_FA_DOTENV={dotenv_path}",  # Pre-resolved .env file path
    ]

    # Also pass through API-related env vars if they're set
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        val = os.environ.get(key)
        if val:
            env_args.append(f"{key}={val}")

    # Build the full sudo command
    cmd = ["sudo", "env"] + env_args + [sys.executable, script] + resolved_args

    # os.execvp replaces the current process entirely with the new command.
    # This function never returns — the current process becomes the sudo process.
    os.execvp("sudo", cmd)


def _resolve_config(config: dict) -> dict:
    """
    Resolve filesystem paths in the config dictionary.

    When running under sudo, ~ expands to /root instead of /home/user.
    To fix this, we resolve cert_dir BEFORE sudo and pass it via _FA_CERT_DIR env var.
    This function uses that pre-resolved path if available.

    Args:
        config: Raw configuration dictionary from YAML

    Returns:
        Config dictionary with cert_dir resolved to an absolute path
    """
    # Prefer the pre-resolved cert_dir from env (set before sudo escalation)
    # Fall back to the value in config.yaml, then to the default
    cert_dir = os.environ.get("_FA_CERT_DIR") or config.get("cert_dir", "~/.free-tools")
    # expanduser() resolves ~, resolve() makes it absolute
    config["cert_dir"] = str(Path(cert_dir).expanduser().resolve())
    return config


# =============================================================================
# COMMAND HANDLERS
# Each function handles one CLI subcommand (start, stop, install-ca, etc.)
# =============================================================================


def cmd_start(args):
    """
    Start the MITM proxy server.

    Execution flow:
    1. Escalate to root (needed for /etc/hosts and port 443)
    2. Load and resolve configuration
    3. Enable DNS spoofing (add entries to /etc/hosts pointing to 127.0.0.1)
    4. Flush DNS cache so changes take effect immediately
    5. Initialize and start the async MITM HTTPS server
    6. On shutdown (Ctrl+C), clean up DNS spoofing entries

    Args:
        args: Parsed argparse namespace (contains args.config path)
    """
    ensure_root()
    config = _resolve_config(load_config(args.config))

    # List of hostnames to intercept (e.g., daily-cloudcode-pa.googleapis.com)
    hosts = config.get("hosts", [])
    # Port to listen on (443 = standard HTTPS, requires root)
    port = config.get("listen_port", 443)

    print(f"[*] Cert dir: {config['cert_dir']}")
    print(
        f"[*] Target: {config.get('target_endpoint') or os.environ.get('ANTHROPIC_BASE_URL', 'NOT SET')}"
    )

    # Add entries to /etc/hosts: "127.0.0.1 <hostname>" for each intercepted host
    print("[*] Enabling DNS spoofing...")
    add_hosts(hosts)
    flush_dns_cache()

    # Create and start the MITM server (handles TLS, request parsing, forwarding)
    print(f"[*] Starting MITM server on port {port}...")
    server = MITMServer(config)

    try:
        # asyncio.run() starts the event loop and blocks until the server stops
        asyncio.run(server.start(port=port))
    except KeyboardInterrupt:
        # User pressed Ctrl+C → graceful shutdown
        print("\n[*] Shutting down...")
    finally:
        # Always clean up DNS entries, even if an error occurred
        print("[*] Disabling DNS spoofing...")
        remove_hosts(hosts)
        flush_dns_cache()
        print("[*] Stopped.")


def cmd_stop(args):
    """
    Stop the MITM proxy by removing DNS spoofing entries.

    This only cleans up /etc/hosts. The actual server process
    should be stopped separately (Ctrl+C or kill).

    Args:
        args: Parsed argparse namespace
    """
    ensure_root()  # Need root to modify /etc/hosts
    config = _resolve_config(load_config(args.config))

    hosts = config.get("hosts", [])

    print("[*] Disabling DNS spoofing...")
    remove_hosts(hosts)
    flush_dns_cache()
    print("[*] Stopped.")


def cmd_install_ca(args):
    """
    Generate (if needed) and install the Root CA into the system trust store.

    This is a one-time setup step. After installation, the system will trust
    certificates signed by our Root CA, allowing the MITM proxy to intercept
    HTTPS traffic without certificate warnings.

    Args:
        args: Parsed argparse namespace
    """
    config = load_config(args.config)
    cert_dir = config.get("cert_dir", "~/.free-tools")

    # install_ca() handles both generation and trust store installation
    success = install_ca(cert_dir)
    sys.exit(0 if success else 1)


def cmd_trust_ca(args):
    """
    Trust the Root CA in the system trust store.

    Platform-specific behavior:
    - Linux (Debian/Ubuntu): Copies cert to /usr/local/share/ca-certificates/
      and runs update-ca-certificates
    - Linux (Fedora/RHEL): Copies to /etc/pki/ca-trust/source/anchors/
      and runs update-ca-trust
    - macOS: Adds to System keychain with trustRoot setting

    Args:
        args: Parsed argparse namespace (may include --force flag)
    """
    config = load_config(args.config)
    cert_dir = config.get("cert_dir", "~/.free-tools")

    # force=True will reinstall even if already trusted
    success = trust_ca(cert_dir, force=args.force)
    sys.exit(0 if success else 1)


def cmd_untrust_ca(args):
    """
    Remove the Root CA from the system trust store.

    The certificate files are NOT deleted — only the trust relationship
    is removed. Use uninstall-ca --delete to also remove the files.

    Args:
        args: Parsed argparse namespace
    """
    config = load_config(args.config)
    cert_dir = config.get("cert_dir", "~/.free-tools")

    success = untrust_ca(cert_dir)
    sys.exit(0 if success else 1)


def cmd_uninstall_ca(args):
    """
    Fully uninstall the Root CA from the system.

    Steps:
    1. Remove from system trust store (untrust)
    2. Optionally delete the certificate files (--delete flag)

    Args:
        args: Parsed argparse namespace (may include --delete flag)
    """
    config = load_config(args.config)
    cert_dir = config.get("cert_dir", "~/.free-tools")

    # delete_files=True will remove rootCA.crt and rootCA.key from disk
    success = uninstall_ca(cert_dir, delete_files=args.delete)
    sys.exit(0 if success else 1)


def cmd_status(args):
    """
    Display the current status of the MITM proxy system.

    Shows:
    - DNS spoofing status (enabled/disabled, which hosts are redirected)
    - Root CA status (exists/not found, trusted/not trusted)
    - Target endpoint configuration
    - Number of model mappings configured

    Args:
        args: Parsed argparse namespace
    """
    config = load_config(args.config)
    hosts = config.get("hosts", [])
    cert_dir = config.get("cert_dir", "~/.free-tools")

    print("=== Free Tools Status ===")
    print()

    # Check DNS spoofing status by looking for our marker block in /etc/hosts
    if is_enabled(hosts):
        print("[*] DNS spoofing: ENABLED")
        print(f"    Redirecting: {', '.join(hosts)}")
    else:
        print("[*] DNS spoofing: DISABLED")

    # Check Root CA certificate status
    print()
    if ca_exists(cert_dir):
        cert_path = get_ca_cert_path(cert_dir)
        print(f"[*] Root CA: {cert_path}")
        trusted = is_trusted(cert_dir)
        print(f"[*] Trusted: {'Yes' if trusted else 'No'}")
    else:
        print("[*] Root CA: Not found (run install-ca first)")

    print()
    print(
        f"Target endpoint: {os.environ.get('ANTHROPIC_BASE_URL', 'Not set (check .env)')}"
    )
    print(f"Model mappings: {len(config.get('model_mapping', {}))} configured")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """
    Main entry point — sets up the CLI argument parser and dispatches commands.

    Available commands:
    - start:        Start the MITM proxy (requires root)
    - stop:         Stop DNS spoofing and clean up /etc/hosts
    - install-ca:   Generate and install Root CA into system trust store
    - trust-ca:     Trust the Root CA (with optional --force)
    - untrust-ca:   Remove Root CA from system trust store
    - uninstall-ca: Full uninstall (with optional --delete to remove files)
    - status:       Show current proxy status

    Global options:
    - -c/--config:  Path to config file (default: config.yaml)
    """
    # Create the top-level argument parser
    parser = argparse.ArgumentParser(
        description="Free Tools - MITM Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global argument: config file path (shared across all subcommands)
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    # Create subcommand parsers — each subcommand has its own handler function
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- start command ---
    start_parser = subparsers.add_parser(
        "start",
        help="Start the MITM proxy",
    )
    start_parser.set_defaults(func=cmd_start)

    # --- stop command ---
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the MITM proxy and clean up DNS",
    )
    stop_parser.set_defaults(func=cmd_stop)

    # --- install-ca command ---
    ca_parser = subparsers.add_parser(
        "install-ca",
        help="Generate and install Root CA into system trust store",
    )
    ca_parser.set_defaults(func=cmd_install_ca)

    # --- trust-ca command (with --force option) ---
    trust_parser = subparsers.add_parser(
        "trust-ca",
        help="Trust the Root CA in the system trust store",
    )
    trust_parser.add_argument(
        "--force", action="store_true", help="Reinstall even if already trusted"
    )
    trust_parser.set_defaults(func=cmd_trust_ca)

    # --- untrust-ca command ---
    untrust_parser = subparsers.add_parser(
        "untrust-ca",
        help="Remove the Root CA from the system trust store",
    )
    untrust_parser.set_defaults(func=cmd_untrust_ca)

    # --- uninstall-ca command (with --delete option) ---
    uninstall_parser = subparsers.add_parser(
        "uninstall-ca",
        help="Uninstall Root CA from trust store and optionally delete files",
    )
    uninstall_parser.add_argument(
        "--delete", action="store_true", help="Also delete CA certificate files"
    )
    uninstall_parser.set_defaults(func=cmd_uninstall_ca)

    # --- status command ---
    status_parser = subparsers.add_parser(
        "status",
        help="Check MITM proxy status",
    )
    status_parser.set_defaults(func=cmd_status)

    # Parse the command-line arguments
    args = parser.parse_args()

    # If no subcommand was given, print help and exit
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to the appropriate command handler function
    args.func(args)


# Standard Python idiom: only run main() when executed directly,
# not when imported as a module
if __name__ == "__main__":
    main()
