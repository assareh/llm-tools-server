# Changelog

## [0.1.0] - 2025-11-21

### Initial Release

Core LLM API Server package extracted from Ivan project.

**Features:**
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, `/health`)
- Support for Ollama and LM Studio backends
- Tool calling with LangChain integration
- System prompt auto-reload
- Open WebUI integration
- Streaming and non-streaming responses
- Debug logging for tool execution

**Supported Backends:**
- Ollama (http://localhost:11434)
- LM Studio (http://localhost:1234/v1)
