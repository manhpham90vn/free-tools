# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Free Antigravity is a MITM (Man-in-the-Middle) proxy that intercepts Gemini API requests from Google's Antigravity Cloud Code extension and forwards them to a configurable custom endpoint (default: Claude API).

## Commands

```bash
# Start the MITM proxy (requires root)
python main.py start

# Stop the proxy
python main.py stop

# Install Root CA into system trust store
python main.py install-ca

# Check proxy status
python main.py status

# Development: lint, format, type check
./scripts/check.sh

# Development: auto-fix lint issues
./scripts/fix.sh
```

## Architecture

### Core Components

- **`main.py`** - CLI entry point with start/stop/status commands
- **`mitm/`** - MITM proxy implementation
  - `server.py` - HTTPS server with SNI-based certificate generation
  - `handler.py` - Request interception and routing (uses `providers/` module)
  - `converter.py` - Legacy conversion logic (being replaced by `providers/`)
  - `passthrough.py` - Forwards non-intercepted requests
  - `cert.py` - SSL certificate generation
- **`providers/`** - Extensible multi-provider adapter module
  - `schema.py` - Internal unified schema (InternalRequest, InternalStreamEvent, etc.)
  - `base.py` - Abstract BaseAdapter class
  - `gemini.py` - Gemini/Antigravity ↔ Internal converter
  - `openai.py` - OpenAI Chat Completions ↔ Internal converter
  - `claude.py` - Claude Messages API ↔ Internal converter
  - `__init__.py` - Registry: `get_adapter(name)`, `register(name, adapter_cls)`
- **`hostsutil/`** - DNS manipulation for /etc/hosts spoofing

### Request Flow

```
Client (Gemini/OpenAI/Claude format)
    ↓
detect_provider(path) → "gemini"/"openai"/"claude"
    ↓
SourceAdapter.parse_request() → InternalRequest
    ↓
TargetAdapter.format_request() → Provider-native format
    ↓
Target API (configurable: Claude, OpenAI, Gemini, etc.)
    ↓
TargetAdapter.parse_stream_event() → InternalStreamEvent
    ↓
SourceAdapter.format_stream_event() → Source-native SSE
    ↓
Client response
```

### Configuration (config.yaml)

```yaml
target_provider: claude  # claude, openai, gemini (default: claude)
target_endpoint: https://api.anthropic.com/v1/messages
api_key: sk-ant-xxx

model_mapping:
  gemini-2.5-flash: claude-opus-4-6

default_model: claude-opus-4-6
```

### Extending with New Providers

To add a new provider (e.g., Ollama):

1. Create `providers/ollama.py` implementing `BaseAdapter`:
```python
class OllamaAdapter(BaseAdapter):
    name = "ollama"

    def parse_request(self, body: bytes, model: str) -> InternalRequest: ...
    def format_request(self, req: InternalRequest) -> dict: ...
    def create_stream_state(self) -> Any: ...
    def parse_stream_event(self, event: dict, state: Any) -> InternalStreamEvent | None: ...
    def format_stream_event(self, event: InternalStreamEvent, state: Any) -> str | None: ...
```

2. Register it:
```python
from providers import register
register("ollama", OllamaAdapter)
```

3. Set `target_provider: ollama` in config.yaml
