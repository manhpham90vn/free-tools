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

from hostsutil.hosts import add_hosts, remove_hosts, flush_dns_cache, is_enabled
from mitm.cert import load_or_create_root_ca
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

    # Ensure CA exists
    ca_key, ca_cert = load_or_create_root_ca(cert_dir)

    cert_path = Path(cert_dir).expanduser() / "rootCA.crt"

    print(f"[*] Root CA location: {cert_path}")
    print()

    # Detect OS and install accordingly
    if sys.platform == "darwin":
        # macOS
        print("On macOS, you can install the CA with:")
        print(
            f"  sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain {cert_path}"
        )
        print()
        print("Or use the GUI:")
        print("  1. Open Keychain Access")
        print(f"  2. Drag {cert_path} to System keychain")
        print("  3. Double-click the certificate, expand Trust")
        print("  4. Set 'Secure Sockets Layer (SSL)' to 'Always Trust'")

    elif sys.platform.startswith("linux"):
        # Linux - varies by distribution
        print("On Linux, you have several options:")
        print()
        print("Option 1 - Debian/Ubuntu:")
        print(f"  sudo cp {cert_path} /usr/local/share/ca-certificates/")
        print("  sudo update-ca-certificates")
        print()
        print("Option 2 - Fedora/RHEL:")
        print(f"  sudo cp {cert_path} /etc/pki/ca-trust/source/anchors/")
        print("  sudo update-ca-trust")
        print()
        print("Option 3 - Chrome (per-user, no sudo needed):")
        print("  1. Open chrome://settings/certificates")
        print("  2. Click 'Authorities' -> 'Import'")
        print(f"  3. Select {cert_path}")
        print("  4. Check 'Trust this certificate for identifying websites'")

    else:
        print(f"Unknown platform: {sys.platform}")
        print(f"Please install {cert_path} manually into your trust store.")

    print()
    print("[*] After installation, restart your browser/application.")


def cmd_status(args):
    """Check the status of the MITM proxy."""
    config = load_config(args.config)
    hosts = config.get("hosts", [])

    print("=== Free Antigravity Status ===")
    print()

    if is_enabled(hosts):
        print("[*] DNS spoofing: ENABLED")
        print(f"    Redirecting: {', '.join(hosts)}")
    else:
        print("[*] DNS spoofing: DISABLED")

    cert_dir = Path(config.get("cert_dir", "~/.free-antigravity")).expanduser()
    if (cert_dir / "rootCA.crt").exists():
        print(f"[*] Root CA: {cert_dir / 'rootCA.crt'}")
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
        help="Install Root CA into system trust store",
    )
    ca_parser.set_defaults(func=cmd_install_ca)

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
