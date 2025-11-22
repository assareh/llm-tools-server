"""Backend communication for Ollama and LM Studio."""

from typing import Any, Dict, List

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
