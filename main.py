#!/usr/bin/env python3
"""
Free Antigravity - MITM Proxy for Antigravity Extension.
Intercepts Gemini API requests and forwards them to a custom endpoint.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from hostsutil import add_hosts, remove_hosts, flush_dns_cache, is_enabled
from cert import (
    install_ca,
    uninstall_ca,
    trust_ca,
    untrust_ca,
    ca_exists,
    is_trusted,
    get_ca_cert_path,
)
from mitm.server import MITMServer

# Load .env file if exists (use env override for sudo, else script directory)
_script_dir = Path(__file__).resolve().parent
_dotenv_path = os.environ.get("_FA_DOTENV", str(_script_dir / ".env"))
load_dotenv(_dotenv_path)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        return yaml.safe_load(f)


def ensure_root():
    """Ensure running as root, using sudo if needed."""
    if os.geteuid() == 0:
        return  # Already root

    # Check if we're already in a sudo session
    if os.environ.get("_FA_SUDO"):
        print("Error: Failed to obtain root privileges.")
        sys.exit(1)

    # Re-execute with sudo
    print("[*] Requesting root privileges via sudo...")

    # Resolve paths and env vars BEFORE sudo (it may strip env for security)
    cert_dir = str(Path("~/.free-antigravity").expanduser().resolve())
    dotenv_path = str((_script_dir / ".env").resolve())

    # Resolve absolute paths before sudo (it doesn't preserve cwd)
    script = str(Path(__file__).resolve())
    cwd = os.getcwd()

    # Rebuild argv with absolute config path
    args = sys.argv[1:]
    resolved_args = []
    i = 0
    config_found = False
    while i < len(args):
        if args[i] in ("-c", "--config") and i + 1 < len(args):
            resolved_args.append(args[i])
            resolved_args.append(str(Path(args[i + 1]).resolve()))
            config_found = True
            i += 2
        else:
            resolved_args.append(args[i])
            i += 1

    # Default config path if not specified
    if not config_found:
        resolved_args = [
            "--config",
            str(Path(cwd, "config.yaml").resolve()),
        ] + resolved_args

    # sudo may strip env vars for security, so use `env` to set them
    # Build: sudo env VAR1=val1 VAR2=val2 python3 script.py ...
    env_args = [
        "_FA_SUDO=1",
        f"_FA_CERT_DIR={cert_dir}",
        f"_FA_DOTENV={dotenv_path}",
    ]

    # Pass through critical env vars
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        val = os.environ.get(key)
        if val:
            env_args.append(f"{key}={val}")

    cmd = ["sudo", "env"] + env_args + [sys.executable, script] + resolved_args
    os.execvp("sudo", cmd)


def _resolve_config(config: dict) -> dict:
    """Resolve paths in config, using env overrides from sudo."""
    # Use cert_dir from env (set before sudo) if available
    cert_dir = os.environ.get("_FA_CERT_DIR") or config.get(
        "cert_dir", "~/.free-antigravity"
    )
    config["cert_dir"] = str(Path(cert_dir).expanduser().resolve())
    return config


def cmd_start(args):
    """Start the MITM proxy."""
    ensure_root()
    config = _resolve_config(load_config(args.config))

    hosts = config.get("hosts", [])
    port = config.get("listen_port", 443)

    print(f"[*] Cert dir: {config['cert_dir']}")
    print(
        f"[*] Target: {config.get('target_endpoint') or os.environ.get('ANTHROPIC_BASE_URL', 'NOT SET')}"
    )

    # Enable DNS spoofing
    print("[*] Enabling DNS spoofing...")
    add_hosts(hosts)
    flush_dns_cache()

    # Start MITM server
    print(f"[*] Starting MITM server on port {port}...")
    server = MITMServer(config)

    try:
        asyncio.run(server.start(port=port))
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    finally:
        # Cleanup DNS
        print("[*] Disabling DNS spoofing...")
        remove_hosts(hosts)
        flush_dns_cache()
        print("[*] Stopped.")


def cmd_stop(args):
    """Stop the MITM proxy and clean up DNS."""
    ensure_root()
    config = _resolve_config(load_config(args.config))

    hosts = config.get("hosts", [])

    print("[*] Disabling DNS spoofing...")
    remove_hosts(hosts)
    flush_dns_cache()
    print("[*] Stopped.")


def cmd_install_ca(args):
    """Install the Root CA into the system trust store."""
    config = load_config(args.config)
    cert_dir = config.get("cert_dir", "~/.free-antigravity")

    # Run install (generate + trust)
    success = install_ca(cert_dir)
    sys.exit(0 if success else 1)


def cmd_trust_ca(args):
    """Trust the Root CA in the system trust store."""
    config = load_config(args.config)
    cert_dir = config.get("cert_dir", "~/.free-antigravity")

    success = trust_ca(cert_dir, force=args.force)
    sys.exit(0 if success else 1)


def cmd_untrust_ca(args):
    """Remove the Root CA from the system trust store."""
    config = load_config(args.config)
    cert_dir = config.get("cert_dir", "~/.free-antigravity")

    success = untrust_ca(cert_dir)
    sys.exit(0 if success else 1)


def cmd_uninstall_ca(args):
    """Uninstall the Root CA from the system and optionally delete files."""
    config = load_config(args.config)
    cert_dir = config.get("cert_dir", "~/.free-antigravity")

    success = uninstall_ca(cert_dir, delete_files=args.delete)
    sys.exit(0 if success else 1)


def cmd_status(args):
    """Check the status of the MITM proxy."""
    config = load_config(args.config)
    hosts = config.get("hosts", [])
    cert_dir = config.get("cert_dir", "~/.free-antigravity")

    print("=== Free Antigravity Status ===")
    print()

    # DNS status
    if is_enabled(hosts):
        print("[*] DNS spoofing: ENABLED")
        print(f"    Redirecting: {', '.join(hosts)}")
    else:
        print("[*] DNS spoofing: DISABLED")

    # CA status
    print()
    if ca_exists(cert_dir):
        cert_path = get_ca_cert_path(cert_dir)
        print(f"[*] Root CA: {cert_path}")
        trusted = is_trusted(cert_dir)
        print(f"[*] Trusted: {'Yes' if trusted else 'No'}")
    else:
        print("[*] Root CA: Not found (run install-ca first)")

    print()
    print(f"Target endpoint: {config.get('target_endpoint', 'Not configured')}")
    print(f"Model mappings: {len(config.get('model_mapping', {}))} configured")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Free Antigravity - MITM Proxy for Antigravity Extension",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start the MITM proxy",
    )
    start_parser.set_defaults(func=cmd_start)

    # stop command
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the MITM proxy and clean up DNS",
    )
    stop_parser.set_defaults(func=cmd_stop)

    # install-ca command
    ca_parser = subparsers.add_parser(
        "install-ca",
        help="Generate and install Root CA into system trust store",
    )
    ca_parser.set_defaults(func=cmd_install_ca)

    # trust-ca command
    trust_parser = subparsers.add_parser(
        "trust-ca",
        help="Trust the Root CA in the system trust store",
    )
    trust_parser.add_argument(
        "--force", action="store_true", help="Reinstall even if already trusted"
    )
    trust_parser.set_defaults(func=cmd_trust_ca)

    # untrust-ca command
    untrust_parser = subparsers.add_parser(
        "untrust-ca",
        help="Remove the Root CA from the system trust store",
    )
    untrust_parser.set_defaults(func=cmd_untrust_ca)

    # uninstall-ca command
    uninstall_parser = subparsers.add_parser(
        "uninstall-ca",
        help="Uninstall Root CA from trust store and optionally delete files",
    )
    uninstall_parser.add_argument(
        "--delete", action="store_true", help="Also delete CA certificate files"
    )
    uninstall_parser.set_defaults(func=cmd_uninstall_ca)

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check MITM proxy status",
    )
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
