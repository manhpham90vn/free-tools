"""
DNS manipulation module for Antigravity MITM proxy.
Manages /etc/hosts entries to redirect traffic to localhost.
"""

import subprocess
from pathlib import Path
from typing import List

ETC_HOSTS = Path("/etc/hosts")
MARKER_START = "# BEGIN ANTIGRAVITY MITM"
MARKER_END = "# END ANTIGRAVITY MITM"


def _read_hosts() -> List[str]:
    """Read /etc/hosts file."""
    return ETC_HOSTS.read_text().splitlines(keepends=True)


def _write_hosts(lines: List[str]) -> None:
    """Write /etc/hosts file."""
    ETC_HOSTS.write_text("".join(lines))


def _run_command(cmd: List[str]) -> None:
    """Run a command. Assumes already running as root (via sudo in main.py)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def add_hosts(hosts: List[str]) -> None:
    """
    Add hosts entries to /etc/hosts to redirect to 127.0.0.1.

    Args:
        hosts: List of hostnames to redirect
    """
    lines = _read_hosts()

    # Remove existing block if present
    lines = _remove_existing_block(lines)

    # Build new block
    block = f"\n{MARKER_START}\n"
    for host in hosts:
        block += f"127.0.0.1 {host}\n"
    block += f"{MARKER_END}\n"

    lines.append(block)
    _write_hosts(lines)

    print(f"Added {len(hosts)} host(s) to /etc/hosts")
    print(
        "Note: You may need to flush DNS cache manually with: resolvectl flush-caches"
    )


def remove_hosts(hosts: List[str]) -> None:
    """
    Remove hosts entries from /etc/hosts.

    Args:
        hosts: List of hostnames to remove
    """
    lines = _read_hosts()
    lines = _remove_existing_block(lines)
    _write_hosts(lines)
    print(f"Removed {len(hosts)} host(s) from /etc/hosts")


def is_enabled(hosts: List[str]) -> bool:
    """
    Check if the hosts entries are currently enabled.

    Args:
        hosts: List of hostnames to check

    Returns:
        True if entries exist, False otherwise
    """
    content = ETC_HOSTS.read_text()
    return MARKER_START in content


def _remove_existing_block(lines: List[str]) -> List[str]:
    """Remove existing Antigravity block from hosts file."""
    result = []
    in_block = False

    for line in lines:
        if MARKER_START in line:
            in_block = True
            continue
        if MARKER_END in line:
            in_block = False
            continue
        if not in_block:
            result.append(line)

    return result


def flush_dns_cache() -> None:
    """Flush system DNS cache."""
    # Try different methods
    methods = [
        ["resolvectl", "flush-caches"],
        ["systemd-resolve", "--flush-caches"],
        ["service", "nscd", "flush"],
    ]

    for cmd in methods:
        try:
            _run_command(cmd)
            print(f"DNS cache flushed via {' '.join(cmd)}")
            return
        except (FileNotFoundError, RuntimeError):
            continue

    print("Warning: Could not flush DNS cache. You may need to flush manually:")
    print("  resolvectl flush-caches")
    print("  or")
    print("  systemd-resolve --flush-caches")
