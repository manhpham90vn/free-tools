# Free Antigravity

A MITM proxy that intercepts requests from Antigravity IDE and forwards them to any LLM provider via custom Anthropic API endpoint.

## 🎯 Goals

Use **Antigravity** with **any LLM provider**:

- **Claude** (Anthropic) - Opus, Sonnet, Haiku
- **OpenAI** - GPT-4, GPT-4o, o1
- **Gemini** (Google) - Gemini 1.5 Pro, Flash
- **Ollama** - Local models (Llama, Mistral, etc.)

Supports importing tokens from multiple sources and automatic protocol translation.

## ✨ Features

- **Multi-Provider Support** - Switch seamlessly between Claude, OpenAI, Gemini, and Ollama
- **Protocol Translation** - Automatically converts between API formats (Gemini ↔ OpenAI ↔ Claude)
- **Streaming Support** - Full SSE streaming for real-time responses
- **Model Mapping** - Map any source model to your preferred target model
- **DNS Spoofing** - Transparent traffic interception via /etc/hosts
- **TLS Interception** - Dynamic certificate generation signed by custom Root CA

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/manhpham90vn/free-antigravity.git
cd free-antigravity

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

Set up API credentials in `.env`:

```bash
cp .env.example .env
```

```env
# Target provider endpoint
ANTHROPIC_BASE_URL=https://api.anthropic.com

# API key
ANTHROPIC_AUTH_TOKEN=sk-ant-...
```

Edit `config.yaml` for behavior settings:

```yaml
target_provider: claude
default_model: "claude-sonnet-4-6"

model_mapping:
  gemini-2.5-pro: claude-opus-4-6
  gemini-2.5-flash: claude-sonnet-4-6
  gpt-4o: claude-sonnet-4-6
```

### 3. Install Root CA

```bash
sudo python main.py setup-ca
```

### 4. Run

```bash
# Start proxy (requires root for port 443 and /etc/hosts)
sudo python main.py start

# Stop proxy
python main.py stop

# Check status
python main.py status
```

## 🛠️ Development

```bash
# Run quality checks
./scripts/check.sh

# Auto-fix issues
./scripts/fix.sh
```

## 📄 License

MIT
