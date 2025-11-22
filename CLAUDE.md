# Claude Assistant Guide - LLM API Server

Developer guide for maintaining the LLM API Server framework.

## Quick Reference

```bash
# Setup (first time)
uv sync --extra dev             # Install all dependencies + dev tools

# Before every commit
./lint.sh                       # Format and lint

# Manual linting
uv run black llm_api_server/    # Format code
uv run ruff check --fix llm_api_server/  # Lint with auto-fix

# Running commands
uv run <command>                # Run any command in the project environment
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
   uv sync  # Will pull llm-api-server from GitHub
   uv run python milesoss.py --no-webui
   ```

2. **Integration testing**: Verify in both Ivan and milesoss

3. **API testing**: Test OpenAI-compatible endpoints
   ```bash
   curl http://localhost:8000/v1/models
   curl http://localhost:8000/health
   ```

## Installation Options

```bash
# Using uv (recommended)
uv sync              # Install core dependencies
uv sync --extra dev  # With development tools
uv sync --extra webui  # With Open Web UI
uv sync --all-extras  # Everything

# Using pip (legacy)
pip install -e .
pip install -e '.[dev]'
pip install -e '.[webui]'
```

## Git Workflow

Standard GitHub workflow:

```bash
# Make changes
./lint.sh  # Format and lint

# Commit
git add .
git commit -m "feat: description

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to GitHub
git push
```

Changes are now distributed via GitHub. Consuming projects install with:
```
llm-api-server @ git+https://github.com/assareh/llm-api-server.git
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
