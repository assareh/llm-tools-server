# LLM API Server

A reusable Flask server providing an OpenAI-compatible API for local LLM backends (Ollama, LM Studio) with tool calling support.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint
- **Multiple backends** - Supports Ollama and LM Studio
- **Tool calling** - Full support for function calling with LangChain tools
- **Streaming responses** - Real-time token streaming
- **WebUI integration** - Optional Open Web UI frontend
- **Smart caching** - System prompt auto-reload on file changes
- **Debug logging** - Comprehensive tool execution logging

## Installation

```bash
pip install llm-api-server
```

Or install from source:

```bash
git clone https://github.com/yourusername/llm-api-server.git
cd llm-api-server
pip install -e .
```

### Optional dependencies

```bash
# For Open Web UI support
pip install llm-api-server[webui]

# For development
pip install llm-api-server[dev]
```

## Quick Start

```python
from llm_api_server import LLMServer, ServerConfig
from langchain_core.tools import tool

# Define your tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

ALL_TOOLS = [get_weather]

# Configure server
config = ServerConfig.from_env("MYAPP_")  # Reads MYAPP_BACKEND, MYAPP_PORT, etc.
config.BACKEND_TYPE = "lmstudio"
config.BACKEND_MODEL = "openai/gpt-oss-20b"
config.MODEL_NAME = "myapp/assistant"
config.SYSTEM_PROMPT_PATH = "system_prompt.md"

# Create and run server
server = LLMServer(
    name="MyApp",
    model_name=config.MODEL_NAME,
    tools=ALL_TOOLS,
    config=config,
    default_system_prompt="You are a helpful assistant."
)

if __name__ == "__main__":
    server.run(port=8000)
```

## Built-in Tools

LLM API Server includes common tools that you can use out of the box:

### Using Built-in Tools

```python
from llm_api_server import LLMServer, BUILTIN_TOOLS, get_current_date, calculate
from langchain_core.tools import tool

# Option 1: Use all built-in tools
server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=BUILTIN_TOOLS,
    config=config
)

# Option 2: Import specific tools
server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=[get_current_date, calculate],
    config=config
)

# Option 3: Combine built-in tools with custom tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=BUILTIN_TOOLS + [get_weather],  # Combine both
    config=config
)
```

### Available Built-in Tools

- **`get_current_date()`** - Returns the current date in YYYY-MM-DD format
- **`calculate(expression: str)`** - Safely evaluates mathematical expressions
  - Supports: `+`, `-`, `*`, `/`, `//`, `%`, `**` (power)
  - Example: `calculate("2 + 3 * 4")` → `"14"`

## Configuration

### Using ServerConfig

```python
from llm_api_server import ServerConfig

# Create from environment variables
config = ServerConfig.from_env("MYAPP_")

# Or configure directly
config = ServerConfig()
config.BACKEND_TYPE = "ollama"  # or "lmstudio"
config.BACKEND_MODEL = "openai/gpt-oss-20b"
config.OLLAMA_ENDPOINT = "http://localhost:11434"
config.DEFAULT_PORT = 8000
config.DEFAULT_TEMPERATURE = 0.0
```

### Environment Variables

With prefix `MYAPP_`:

- `MYAPP_BACKEND` - Backend type (ollama, lmstudio)
- `MYAPP_BACKEND_MODEL` - Model identifier
- `MYAPP_PORT` - Server port (default: 8000)
- `MYAPP_TEMPERATURE` - Default temperature (default: 0.0)
- `MYAPP_SYSTEM_PROMPT_PATH` - Path to system prompt file
- `MYAPP_DEBUG_TOOLS` - Enable tool debug logging (true/false)
- `OLLAMA_ENDPOINT` - Ollama API endpoint
- `LMSTUDIO_ENDPOINT` - LM Studio API endpoint

## API Endpoints

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

### `GET /v1/models`

List available models.

```bash
curl http://localhost:8000/v1/models
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "myapp/assistant",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "stream": false
  }'
```

## Advanced Usage

### Custom Initialization Hook

```python
def init_database():
    print("Initializing database...")
    # Your initialization code here

server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=ALL_TOOLS,
    config=config,
    init_hook=init_database  # Called before server starts
)
```

### Custom Logger Names

```python
server = LLMServer(
    name="MyApp",
    model_name="myapp/assistant",
    tools=ALL_TOOLS,
    config=config,
    logger_names=["myapp.tools", "myapp.backend", "tools"]
)
```

### System Prompt Auto-Reload

Create a `system_prompt.md` file:

```markdown
You are MyApp, an AI assistant specialized in...
```

The server automatically reloads this file when it changes (based on modification time).

## Examples

See the `examples/` directory for complete examples:

- **Ivan** - HashiCorp documentation expert
- **Miles** - Credit card rewards optimizer

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-api-server.git
cd llm-api-server

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black llm_api_server/

# Lint
flake8 llm_api_server/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Projects Using LLM API Server

- **[Ivan](https://github.com/assareh/ivan)** - AI assistant for HashiCorp solutions engineers
- **[Miles OSS](https://github.com/assareh/milesoss)** - Credit card rewards optimization assistant

## Acknowledgments

Built with:
- [Flask](https://flask.palletsprojects.com/)
- [LangChain](https://www.langchain.com/)
- [Open Web UI](https://github.com/open-webui/open-webui)
