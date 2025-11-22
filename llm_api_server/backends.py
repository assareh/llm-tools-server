"""Backend communication for Ollama and LM Studio."""

from typing import Any, Dict, List, Tuple

import requests


def get_tool_schema(tool) -> Dict[str, Any]:
    """Extract schema from LangChain tool (handles both pydantic v1 and v2)."""
    if hasattr(tool.args_schema, "model_json_schema"):
        return tool.args_schema.model_json_schema()
    elif hasattr(tool.args_schema, "schema"):
        return tool.args_schema.schema()
    return {}


def call_ollama(messages: List[Dict], tools: List, config, temperature: float = 0.0, stream: bool = False):
    """Call Ollama with tool support."""
    endpoint = f"{config.OLLAMA_ENDPOINT}/api/chat"

    # Convert tools to Ollama format
    ollama_tools = []
    for tool in tools:
        schema = get_tool_schema(tool)
        tool_def = {
            "type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": schema},
        }
        ollama_tools.append(tool_def)

    payload = {
        "model": config.BACKEND_MODEL,
        "messages": messages,
        "tools": ollama_tools,
        "stream": stream,
        "options": {"temperature": temperature},
    }

    # Set timeout as tuple (connect_timeout, read_timeout)
    timeout = (config.BACKEND_CONNECT_TIMEOUT, config.BACKEND_READ_TIMEOUT)

    response = requests.post(endpoint, json=payload, stream=stream, timeout=timeout)
    response.raise_for_status()
    return response


def call_lmstudio(messages: List[Dict], tools: List, config, temperature: float = 0.0, stream: bool = False):
    """Call LM Studio with tool support."""
    endpoint = f"{config.LMSTUDIO_ENDPOINT}/chat/completions"

    # Convert tools to OpenAI format
    openai_tools = []
    for tool in tools:
        schema = get_tool_schema(tool)
        tool_def = {
            "type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": schema},
        }
        openai_tools.append(tool_def)

    payload = {
        "model": config.BACKEND_MODEL,
        "messages": messages,
        "tools": openai_tools,
        "temperature": temperature,
        "stream": stream,
    }

    # Set timeout as tuple (connect_timeout, read_timeout)
    timeout = (config.BACKEND_CONNECT_TIMEOUT, config.BACKEND_READ_TIMEOUT)

    response = requests.post(endpoint, json=payload, stream=stream, timeout=timeout)
    response.raise_for_status()
    return response


def check_ollama_health(config, timeout: int = 5) -> Tuple[bool, str]:
    """Check if Ollama backend is healthy and reachable.

    Args:
        config: ServerConfig instance
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_healthy: bool, message: str)
    """
    try:
        endpoint = f"{config.OLLAMA_ENDPOINT}/api/tags"
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()

        # Check if the configured model is available
        data = response.json()
        models = data.get("models", [])
        model_names = [model.get("name", "") for model in models]

        if config.BACKEND_MODEL in model_names:
            return True, f"Ollama is healthy. Model '{config.BACKEND_MODEL}' is available."
        else:
            available = ", ".join(model_names) if model_names else "none"
            return (
                False,
                f"Ollama is reachable but model '{config.BACKEND_MODEL}' not found. Available models: {available}",
            )

    except requests.Timeout:
        return False, f"Ollama health check timed out after {timeout}s. Backend may be unresponsive."
    except requests.ConnectionError:
        return False, f"Cannot connect to Ollama at {config.OLLAMA_ENDPOINT}. Is it running?"
    except Exception as e:
        return False, f"Ollama health check failed: {e!s}"


def check_lmstudio_health(config, timeout: int = 5) -> Tuple[bool, str]:
    """Check if LM Studio backend is healthy and reachable.

    Args:
        config: ServerConfig instance
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_healthy: bool, message: str)
    """
    try:
        endpoint = f"{config.LMSTUDIO_ENDPOINT}/models"
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()

        # LM Studio returns model list if healthy
        data = response.json()
        models = data.get("data", [])

        if models:
            model_ids = [model.get("id", "") for model in models]
            return True, f"LM Studio is healthy. {len(models)} model(s) loaded: {', '.join(model_ids)}"
        else:
            return False, "LM Studio is reachable but no models are loaded. Please load a model in LM Studio."

    except requests.Timeout:
        return False, f"LM Studio health check timed out after {timeout}s. Backend may be unresponsive."
    except requests.ConnectionError:
        return False, f"Cannot connect to LM Studio at {config.LMSTUDIO_ENDPOINT}. Is it running?"
    except Exception as e:
        return False, f"LM Studio health check failed: {e!s}"
