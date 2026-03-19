"""
DNS Manipulation Module for Free Tools MITM Proxy.

This module manages /etc/hosts entries to redirect traffic to localhost.

How DNS spoofing works:
1. We add entries to /etc/hosts mapping target hostnames to 127.0.0.1
2. When the client (IDE, Cursor, etc.) tries to connect to the API,
   the OS resolves the hostname to 127.0.0.1 (our proxy)
3. Our MITM proxy receives the connection and handles it

The entries are wrapped in marker comments (BEGIN/END FREE TOOLS MITM)
so we can cleanly add and remove them without affecting other /etc/hosts entries.

Example /etc/hosts block:
    # BEGIN FREE TOOLS MITM
    127.0.0.1 daily-cloudcode-pa.googleapis.com
    127.0.0.1 cloudcode-pa.googleapis.com
    # END FREE TOOLS MITM
"""

# === Standard library imports ===
import subprocess  # For running system commands (DNS cache flush)
from pathlib import Path  # For filesystem operations
from typing import List  # Type hints

# === Internal module imports ===
from logger import get_logger  # Structured logging

# Module-level logger
log = get_logger("hostsutil")

# Path to the system hosts file
ETC_HOSTS = Path("/etc/hosts")

# Marker comments used to identify our block in /etc/hosts
# These allow us to cleanly add/remove entries without affecting other entries
MARKER_START = "# BEGIN FREE TOOLS MITM"
MARKER_END = "# END FREE TOOLS MITM"


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _read_hosts() -> List[str]:
    """
    Read the /etc/hosts file and return its lines.

    Returns:
        List of lines from /etc/hosts (with newlines preserved)
    """
    return ETC_HOSTS.read_text().splitlines(keepends=True)


def _write_hosts(lines: List[str]) -> None:
    """
    Write lines back to /etc/hosts.

    Args:
        lines: List of lines to write (should include newlines)
    """
    ETC_HOSTS.write_text("".join(lines))


def _run_command(cmd: List[str]) -> None:
    """
    Run a system command. Assumes already running as root.

    Args:
        cmd: Command and arguments as a list

    Raises:
        RuntimeError: If the command fails (non-zero exit code)
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def _remove_existing_block(lines: List[str]) -> List[str]:
    """
    Remove the existing Free Tools block from /etc/hosts lines.

    Scans through the lines and removes everything between
    MARKER_START and MARKER_END (inclusive).

    Args:
        lines: Current lines from /etc/hosts

    Returns:
        Lines with the Free Tools block removed
    """
    result = []
    in_block = False

    for line in lines:
        # Start of our block - skip this line and set flag
        if MARKER_START in line:
            in_block = True
            continue
        # End of our block - skip this line and clear flag
        if MARKER_END in line:
            in_block = False
            continue
        # Only keep lines that are NOT inside our block
        if not in_block:
            result.append(line)

    return result


# =============================================================================
# PUBLIC API
# =============================================================================


def add_hosts(hosts: List[str]) -> None:
    """
    Add host entries to /etc/hosts to redirect traffic to 127.0.0.1.

    This is the core of DNS spoofing. Each hostname gets an entry
    pointing to 127.0.0.1 (localhost), so when the client tries to
    connect to the API, it connects to our MITM proxy instead.

    The entries are wrapped in marker comments for easy removal.

    Args:
        hosts: List of hostnames to redirect (e.g., ["cloudcode-pa.googleapis.com"])
    """
    lines = _read_hosts()

    # Remove existing block if present (idempotent operation)
    lines = _remove_existing_block(lines)

    # Build new block with marker comments
    block = f"\n{MARKER_START}\n"
    for host in hosts:
        block += f"127.0.0.1 {host}\n"
    block += f"{MARKER_END}\n"

    # Append to hosts file
    lines.append(block)
    _write_hosts(lines)

    log.info("Added {n} host(s) to /etc/hosts", n=len(hosts))
    log.info(
        "Note: You may need to flush DNS cache manually with: resolvectl flush-caches"
    )


def remove_hosts(hosts: List[str]) -> None:
    """
    Remove our host entries from /etc/hosts.

    Removes the entire Free Tools block (between markers).
    Other entries in /etc/hosts are not affected.

    Args:
        hosts: List of hostnames (used only for logging)
    """
    lines = _read_hosts()
    lines = _remove_existing_block(lines)
    _write_hosts(lines)
    log.info("Removed {n} host(s) from /etc/hosts", n=len(hosts))


def is_enabled(hosts: List[str]) -> bool:
    """
    Check if DNS spoofing is currently enabled.

    Looks for our marker comment in /etc/hosts.

    Args:
        hosts: List of hostnames (not used, kept for API consistency)

    Returns:
        True if our entries exist in /etc/hosts
    """
    content = ETC_HOSTS.read_text()
    return MARKER_START in content


def flush_dns_cache() -> None:
    """
    Flush the system DNS cache so /etc/hosts changes take effect immediately.

    Tries multiple methods since different Linux distributions use
    different DNS cache managers:
    1. resolvectl flush-caches (systemd-resolved, modern Ubuntu/Fedora)
    2. systemd-resolve --flush-caches (older systemd)
    3. service nscd flush (nscd daemon)

    If none work, prints instructions for manual flushing.
    """
    methods = [
        ["resolvectl", "flush-caches"],
        ["systemd-resolve", "--flush-caches"],
        ["service", "nscd", "flush"],
    ]

    for cmd in methods:
        try:
            _run_command(cmd)
            log.info("DNS cache flushed via {cmd}", cmd=" ".join(cmd))
            return
        except (FileNotFoundError, RuntimeError):
            continue  # Try next method

    # None of the methods worked
    log.warning("Could not flush DNS cache. You may need to flush manually:")
    log.warning("  resolvectl flush-caches  or  systemd-resolve --flush-caches")
