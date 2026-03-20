"""
Free Antigravity Logger.

A structured, color-coded logging module built on top of Python's stdlib `logging`.
Replaces ad-hoc `print()` calls throughout the codebase with typed, configurable loggers.

Usage:
    from logger import get_logger, setup_logging, LogLevel

    setup_logging(level=LogLevel.INFO)
    log = get_logger("mitm.server")
    log.info("Server started on port {port}", port=443)
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from enum import Enum, auto
from pathlib import Path
from typing import Any, TextIO

# =============================================================================
# Constants & Palette
# =============================================================================

# ANSI escape codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Palette entry: (fg_code, bold?)
PaletteEntry = tuple[str, bool]

# Log-level colour pairs
_P_DEBUG: PaletteEntry = ("\033[36m", False)  # cyan
_P_INFO: PaletteEntry = ("\033[32m", False)  # green
_P_WARNING: PaletteEntry = ("\033[33m", True)  # yellow bold
_P_ERROR: PaletteEntry = ("\033[31m", True)  # red bold
_P_CRITICAL: PaletteEntry = ("\033[35m", True)  # magenta bold
_P_BANNER: PaletteEntry = ("\033[94m", True)  # blue bold
_P_SUCCESS: PaletteEntry = ("\033[32m", True)  # green bold
_P_STREAM: PaletteEntry = ("\033[34m", False)  # blue
_P_SNI: PaletteEntry = ("\033[90m", False)  # dim grey
_P_INTERCEPT: PaletteEntry = ("\033[35m", False)  # magenta
_P_REQ: PaletteEntry = ("\033[90m", False)  # dim grey
_P_RESP: PaletteEntry = ("\033[90m", False)  # dim grey


def _palette_for_level(level: int) -> PaletteEntry:
    if level >= logging.CRITICAL:
        return _P_CRITICAL
    if level >= logging.ERROR:
        return _P_ERROR
    if level >= logging.WARNING:
        return _P_WARNING
    if level >= logging.INFO:
        return _P_INFO
    return _P_DEBUG


# Convenience prefixes
PREFIX_STAR = "[*]"
PREFIX_PLUS = "[+]"
PREFIX_WARN = "[!]"
PREFIX_ERROR = "[ERROR]"
PREFIX_INFO = "[INFO]"


# =============================================================================
# LogLevel enum
# =============================================================================


class LogLevel(Enum):
    """Application-level log level enum — maps to stdlib logging levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    STREAM = auto()
    SNI = auto()
    INTERCEPT = auto()
    REQ = auto()
    RESP = auto()

    @property
    def lib_level(self) -> int:
        """Map to stdlib logging level; custom levels clamp to DEBUG."""
        if self in {
            LogLevel.STREAM,
            LogLevel.SNI,
            LogLevel.INTERCEPT,
            LogLevel.REQ,
            LogLevel.RESP,
        }:
            return logging.DEBUG
        return self.value

    @classmethod
    def from_string(cls, name: str) -> LogLevel:
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown log level: {name!r}. Valid: {[e.name for e in cls]}"
            ) from None


# =============================================================================
# Colouring formatter
# =============================================================================


class _ColouredFormatter(logging.Formatter):
    """Adds ANSI colour codes to log records based on their level."""

    def __init__(
        self, fmt: str, datefmt: str | None = None, *, use_color: bool = True
    ) -> None:
        super().__init__(fmt, datefmt)
        self._use_color = use_color

    def _colour(self, level: int, text: str) -> str:
        if not self._use_color:
            return text
        fg, bold = _palette_for_level(level)
        bold_code = BOLD if bold else ""
        return f"{fg}{bold_code}{text}{RESET}"

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = self._colour(record.levelno, record.levelname.ljust(8))
        return super().format(record)


# =============================================================================
# Colouring stream handler (mirrors print() prefixes)
# =============================================================================


_NAMESPACE_PREFIXES: dict[str, tuple[str, PaletteEntry]] = {
    "stream": ("[STREAM]", _P_STREAM),
    "sni": ("[SNI]", _P_SNI),
    "intercept": ("[INTERCEPT]", _P_INTERCEPT),
    "req": ("[REQ]", _P_REQ),
    "resp": ("[RESP]", _P_RESP),
    "banner": ("[*]", _P_BANNER),
    "success": ("[+]", _P_SUCCESS),
    "passthrough": ("[PASSTHROUGH]", _P_WARNING),
    "loop": ("[LOOP]", _P_WARNING),
}


class _PrefixHandler(logging.Handler):
    """
    Logging handler that mirrors the original print()-style prefixes:
        [*]  → INFO     (green)
        [!]  → WARNING  (yellow)
        [ERROR] → ERROR  (red bold)
    """

    def __init__(self, stream: TextIO | None = None, *, use_color: bool = True) -> None:
        super().__init__()
        self._stream: TextIO = stream or sys.stdout
        self._use_color = use_color

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._stream.write(msg + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def format(self, record: logging.LogRecord) -> str:
        prefix, palette = self._resolve_prefix(record)
        text = self._colour(f"{prefix}  {record.getMessage()}", palette)
        if record.exc_info:
            formatter = logging.Formatter()
            text += "\n" + formatter.formatException(record.exc_info)
        return text

    def _resolve_prefix(self, record: logging.LogRecord) -> tuple[str, PaletteEntry]:
        ns = getattr(record, "_namespace", None)
        if ns and ns in _NAMESPACE_PREFIXES:
            return _NAMESPACE_PREFIXES[ns]

        if record.levelno >= logging.CRITICAL:
            return PREFIX_ERROR, _P_CRITICAL
        if record.levelno >= logging.ERROR:
            return PREFIX_ERROR, _P_ERROR
        if record.levelno >= logging.WARNING:
            return PREFIX_WARN, _P_WARNING
        if record.levelno >= logging.INFO:
            return PREFIX_STAR, _P_INFO
        return PREFIX_INFO, _P_DEBUG

    def _colour(self, text: str, palette_entry: PaletteEntry) -> str:
        if not self._use_color:
            return text
        fg, bold = palette_entry
        bold_code = BOLD if bold else ""
        return f"{fg}{bold_code}{text}{RESET}"


# =============================================================================
# File handler with rotation
# =============================================================================


class _RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating handler that writes plain (non-colour) log lines to a file."""

    def __init__(
        self,
        filename: str | Path,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            str(filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            **kwargs,
        )

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = record.levelname.strip()
        return super().format(record)


# =============================================================================
# Main Logger class
# =============================================================================


class FreeLogger:
    """
    Application logger wrapper around Python's stdlib `logging.Logger`.

    Provides:
    - Structured message templates via `.info("msg {x}", x=1)`
    - Custom namespace shortcuts for coloured prefixes
    - Seamless filtering to a configured minimum level
    """

    def __init__(self, lib_logger: logging.Logger) -> None:
        self._lib_logger = lib_logger

    # Standard logging methods

    def debug(self, msg: str, /, **kwargs: Any) -> None:
        self._lib_logger.debug(msg.format(**kwargs) if kwargs else msg)

    def info(self, msg: str, /, **kwargs: Any) -> None:
        self._lib_logger.info(msg.format(**kwargs) if kwargs else msg)

    def warning(self, msg: str, /, **kwargs: Any) -> None:
        self._lib_logger.warning(msg.format(**kwargs) if kwargs else msg)

    def error(self, msg: str, /, **kwargs: Any) -> None:
        self._lib_logger.error(msg.format(**kwargs) if kwargs else msg)

    def critical(self, msg: str, /, **kwargs: Any) -> None:
        self._lib_logger.critical(msg.format(**kwargs) if kwargs else msg)

    def exception(self, msg: str, /, **kwargs: Any) -> None:
        self._lib_logger.exception(msg.format(**kwargs) if kwargs else msg)

    # Custom namespace methods

    def banner(self, msg: str, /, **kwargs: Any) -> None:
        """Log a banner/startup message  → [*]  (blue bold)."""
        self._log_with_ns("banner", logging.INFO, msg, kwargs)

    def success(self, msg: str, /, **kwargs: Any) -> None:
        """Log a success message  → [+]  (green bold)."""
        self._log_with_ns("success", logging.INFO, msg, kwargs)

    def stream(self, msg: str, /, **kwargs: Any) -> None:
        """Log an SSE / streaming event  → [STREAM]  (blue)."""
        self._log_with_ns("stream", logging.DEBUG, msg, kwargs)

    def sni(self, msg: str, /, **kwargs: Any) -> None:
        """Log SNI / TLS handshake details  → [SNI]  (dim grey)."""
        self._log_with_ns("sni", logging.DEBUG, msg, kwargs)

    def intercept(self, msg: str, /, **kwargs: Any) -> None:
        """Log an intercepted request  → [INTERCEPT]  (magenta)."""
        self._log_with_ns("intercept", logging.DEBUG, msg, kwargs)

    def req(self, msg: str, /, **kwargs: Any) -> None:
        """Log raw request parsing  → [REQ]  (dim grey)."""
        self._log_with_ns("req", logging.DEBUG, msg, kwargs)

    def resp(self, msg: str, /, **kwargs: Any) -> None:
        """Log raw response details  → [RESP]  (dim grey)."""
        self._log_with_ns("resp", logging.DEBUG, msg, kwargs)

    def passthrough(self, msg: str, /, **kwargs: Any) -> None:
        """Log a passthrough (non-intercepted) request  → [PASSTHROUGH]."""
        self._log_with_ns("passthrough", logging.INFO, msg, kwargs)

    def loop(self, msg: str, /, **kwargs: Any) -> None:
        """Log a loop-detection / self-request drop  → [LOOP]."""
        self._log_with_ns("loop", logging.INFO, msg, kwargs)

    # Internal

    def _log_with_ns(
        self, namespace: str, fallback_level: int, msg: str, kwargs: dict[str, Any]
    ) -> None:
        formatted = msg.format(**kwargs) if kwargs else msg
        record = self._lib_logger.makeRecord(
            self._lib_logger.name,
            fallback_level,
            "(unknown)",
            0,
            formatted,
            (),
            None,
        )
        record._namespace = namespace  # type: ignore[attr-defined]
        self._lib_logger.handle(record)


# =============================================================================
# Global logger registry  (singleton pattern)
# =============================================================================


class _LoggerRegistry:
    """Singleton registry that manages the shared logging configuration."""

    def __init__(self) -> None:
        self._root = logging.getLogger("free-antigravity")
        self._configured = False
        self._use_color = True
        self._file_handler: logging.Handler | None = None

    def get_logger(self, name: str) -> FreeLogger:
        lib = logging.getLogger(f"free-antigravity.{name.lstrip('free-antigravity.')}")
        return FreeLogger(lib)

    def setup(
        self,
        level: LogLevel = LogLevel.INFO,
        *,
        log_file: str | Path | None = None,
        use_color: bool = True,
        force: bool = False,
    ) -> None:
        if self._configured and not force:
            return

        self._use_color = use_color

        if use_color and sys.stdout.isatty() is False:
            use_color = False

        effective = level.lib_level
        self._root.setLevel(effective)

        for h in self._root.handlers[:]:
            self._root.removeHandler(h)

        console = _PrefixHandler(use_color=use_color)
        console.setLevel(effective)
        console.setFormatter(_ColouredFormatter("%(message)s", use_color=use_color))
        self._root.addHandler(console)

        if log_file:
            log_path = Path(log_file).expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = _RotatingFileHandler(log_path)
            file_handler.setLevel(effective)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._root.addHandler(file_handler)
            self._file_handler = file_handler

        self._configured = True

    def is_configured(self) -> bool:
        return self._configured

    @property
    def color_enabled(self) -> bool:
        return self._use_color


# Module-level singleton
_registry = _LoggerRegistry()


# =============================================================================
# Public API
# =============================================================================


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    *,
    log_file: str | Path | None = None,
    use_color: bool = True,
    force: bool = False,
) -> None:
    """
    One-time setup for the global logging subsystem.

    Call this once, early in ``main.py``, before any other module logs.
    """
    _registry.setup(level, log_file=log_file, use_color=use_color, force=force)


def get_logger(name: str = "") -> FreeLogger:
    """
    Return a :class:`FreeLogger` for the given module name.

    Usage::

        log = get_logger("mitm.server")
        log.info("Listening on port {port}", port=443)
        log.stream("Event type: {t}", t="message")
    """
    return _registry.get_logger(name)


def add_argument_group(parser: argparse.ArgumentParser) -> None:
    """Add ``--log-level``, ``--log-file``, and ``--no-color`` to an ArgumentParser."""
    group = parser.add_argument_group("Logging")
    group.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Minimum log level (debug, info, warning, error). Default: info.",
    )
    group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write logs to this file (rotating, 10 MB max, 5 backups).",
    )
    group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour codes in log output.",
    )


def parse_log_args(
    args: argparse.Namespace,
) -> tuple[LogLevel, str | Path | None, bool]:
    """Parse log-related arguments added by :func:`add_argument_group`."""
    level_str = getattr(args, "log_level", "info")
    try:
        level = LogLevel.from_string(level_str)
    except ValueError:
        level = LogLevel.INFO
    log_file: str | Path | None = getattr(args, "log_file", None)
    use_color = not getattr(args, "no_color", False)
    return level, log_file, use_color
