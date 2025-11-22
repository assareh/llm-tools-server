# Claude Assistant Guide - LLM API Server

Developer guide for maintaining the LLM API Server framework.

## Quick Reference

```bash
# Before every commit
./lint.sh

# Manual linting
black llm_api_server/          # Format code
ruff check --fix llm_api_server/  # Lint with auto-fix

# Install for development
pip install -e '.[dev]'         # Editable install with dev tools
```

## Project Overview

LLM API Server is a reusable Flask framework for building OpenAI-compatible API servers with:
- LM Studio and Ollama backend support
- LangChain tool calling integration
- Open Web UI integration
- Extensible configuration system

## Linting Routine

### Standard Workflow

Always run before committing:

```bash
./lint.sh
```

This script:
1. Formats code with Black (120 char lines)
2. Lints with Ruff (auto-fixes most issues)
3. Verifies all checks pass

### Configuration

All linting settings in `pyproject.toml`:
- **Black**: 120 character lines, Python 3.8-3.12 support
- **Ruff**: Fast linter replacing flake8/isort/pylint
- **MyPy**: Optional type checking

## Development Guidelines

### Code Style
- Line length: 120 characters max
- Python version: 3.8+ compatibility
- Type hints: Use modern syntax where possible
- Imports: Auto-sorted by Ruff (stdlib → third-party → first-party)

### Package Structure

```
llm-api-server/
├── llm_api_server/
│   ├── __init__.py       # Package exports
│   ├── server.py         # Core LLMServer class
│   ├── backends.py       # Backend integrations
│   ├── config.py         # ServerConfig base class
│   └── webui.py          # Open Web UI integration
├── setup.py              # Package installation
├── pyproject.toml        # Packaging & linting config
└── README.md             # Package documentation
```

### Making Changes

1. **Core server** (`server.py`): Flask app, routing, tool calling loop
2. **Backends** (`backends.py`): Ollama/LM Studio communication
3. **Config** (`config.py`): Configuration and environment loading
4. **Web UI** (`webui.py`): Open Web UI subprocess management

### Adding Features

When adding new features, consider:
- **Backwards compatibility**: This is used by multiple projects
- **Configuration options**: Make features configurable
- **Documentation**: Update README.md and docstrings
- **Examples**: Update consuming projects (Ivan, milesoss)

## Testing

Since this is a framework library:

1. **Local testing**: Install in consuming project
   ```bash
   cd ../milesoss  # or ../Ivan
   pip install -e ../llm-api-server
   python milesoss.py --no-webui
   ```

2. **Integration testing**: Verify in both Ivan and milesoss

3. **API testing**: Test OpenAI-compatible endpoints
   ```bash
   curl http://localhost:8000/v1/models
   curl http://localhost:8000/health
   ```

## Installation Options

```bash
# Editable install (development)
pip install -e .

# With development tools
pip install -e '.[dev]'

# With Open Web UI
pip install -e '.[webui]'

# From git (in consuming projects)
pip install -e ../llm-api-server
```

## Git Workflow

Since this is not currently a git repository, changes are distributed via:
- Direct file editing
- Reinstallation in consuming projects: `pip install -e ../llm-api-server --force-reinstall`

If converting to git:
```bash
git init
git add .
git commit -m "feat: description

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Key Components

### LLMServer Class
Main server class that consuming projects instantiate:
```python
server = LLMServer(
    name="MyApp",
    model_name="myapp/model",
    tools=ALL_TOOLS,
    config=config,
    default_system_prompt="You are...",
    init_hook=initialization_function,
    logger_names=["myapp.tools"]
)
```

### ServerConfig Base Class
Extensible configuration loaded from environment:
```python
class MyConfig(ServerConfig):
    CUSTOM_SETTING: str = "default"

config = ServerConfig.from_env("MYAPP_")
```

### Backend Support
- **Ollama**: Native Ollama API format
- **LM Studio**: OpenAI-compatible format
- **Tool calling**: Automatic conversion and execution

## Consuming Projects

Current projects using this framework:
- **Ivan**: HashiCorp documentation assistant
- **milesoss**: Credit card rewards optimizer

When making changes, test in both projects.

## Resources

- [README.md](README.md) - Package documentation
- [Black Docs](https://black.readthedocs.io/)
- [Ruff Docs](https://docs.astral.sh/ruff/)
- [Flask](https://flask.palletsprojects.com/)
- [LangChain](https://python.langchain.com/)

---

*Last updated: 2025-11-22*
*Version: 0.1.0*
