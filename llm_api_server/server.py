"""Core LLM API Server implementation."""

import json
import logging
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from .backends import call_lmstudio, call_ollama, check_lmstudio_health, check_ollama_health
from .config import ServerConfig


class LLMServer:
    """Flask server providing OpenAI-compatible API for LLM backends with tool calling."""

    def __init__(
        self,
        name: str,
        model_name: str,
        tools: List,
        config: ServerConfig,
        default_system_prompt: str = "You are a helpful AI assistant.",
        init_hook: Optional[Callable] = None,
        logger_names: Optional[List[str]] = None,
    ):
        """Initialize LLM API Server.

        Args:
            name: Display name for the server (e.g., "Ivan", "Miles")
            model_name: Model identifier to advertise (e.g., "wwtfo/ivan")
            tools: List of LangChain tools
            config: ServerConfig instance
            default_system_prompt: Default system prompt if file doesn't exist
            init_hook: Optional function to call during initialization (e.g., index building)
            logger_names: Optional list of logger names for debug logging
        """
        self.name = name
        self.model_name = model_name
        self.tools = tools
        self.config = config
        self.default_system_prompt = default_system_prompt
        self.init_hook = init_hook

        # System prompt caching
        self._system_prompt_cache: Optional[str] = None
        self._system_prompt_mtime: Optional[float] = None

        # WebUI process
        self._webui_process = None

        # Create Flask app
        self.app = Flask(name.lower())
        CORS(self.app)

        # Configure logging
        self.logger = logging.getLogger(f"{name.lower()}.tools")
        logger_names = logger_names or [f"{name.lower()}.tools", "tools"]

        if config.DEBUG_TOOLS:
            log_file = Path(config.DEBUG_TOOLS_LOG_FILE)
            # Use RotatingFileHandler for automatic log rotation
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.DEBUG_LOG_MAX_BYTES,
                backupCount=config.DEBUG_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)

            for logger_name in logger_names:
                logger_obj = logging.getLogger(logger_name)
                logger_obj.setLevel(logging.DEBUG)
                logger_obj.addHandler(file_handler)

            max_mb = config.DEBUG_LOG_MAX_BYTES / (1024 * 1024)
            print(f"Tool debug logging enabled: {log_file.absolute()}")
            print(f"  Logging: {', '.join(logger_names)}")
            print(f"  Rotation: {max_mb:.1f}MB max, {config.DEBUG_LOG_BACKUP_COUNT} backups")
        else:
            self.logger.setLevel(logging.WARNING)

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register Flask routes."""
        self.app.route("/health", methods=["GET"])(self.health)
        self.app.route("/v1/models", methods=["GET"])(self.list_models)
        self.app.route("/v1/chat/completions", methods=["POST"])(self.chat_completions)

    def get_system_prompt(self) -> str:
        """Load system prompt from markdown file with smart caching."""
        prompt_path = Path(self.config.SYSTEM_PROMPT_PATH)

        if not prompt_path.exists():
            return self.default_system_prompt

        try:
            current_mtime = prompt_path.stat().st_mtime

            # Check if cache is valid
            if self._system_prompt_cache is not None and self._system_prompt_mtime == current_mtime:
                return self._system_prompt_cache

            # Read and cache the prompt
            self._system_prompt_cache = prompt_path.read_text(encoding="utf-8")
            self._system_prompt_mtime = current_mtime

            return self._system_prompt_cache
        except Exception as e:
            print(f"Error reading system prompt: {e}")
            return self.default_system_prompt

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool by name with given input."""
        # Log tool call
        if self.config.DEBUG_TOOLS:
            self.logger.debug("=" * 80)
            self.logger.debug(f"TOOL CALL: {tool_name}")
            self.logger.debug(f"INPUT: {json.dumps(tool_input, indent=2)}")

        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.func(**tool_input)
                    result_str = str(result)

                    # Log tool response
                    if self.config.DEBUG_TOOLS:
                        if len(result_str) > 1000:
                            truncated = result_str[:1000] + f"\n... (truncated, total length: {len(result_str)} chars)"
                            self.logger.debug(f"RESPONSE: {truncated}")
                        else:
                            self.logger.debug(f"RESPONSE: {result_str}")
                        self.logger.debug("=" * 80 + "\n")

                    return result_str
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {e!s}"
                    if self.config.DEBUG_TOOLS:
                        self.logger.error(f"ERROR: {error_msg}")
                        self.logger.debug("=" * 80 + "\n")
                    return error_msg

        not_found_msg = f"Tool {tool_name} not found"
        if self.config.DEBUG_TOOLS:
            self.logger.error(f"ERROR: {not_found_msg}")
            self.logger.debug("=" * 80 + "\n")
        return not_found_msg

    def call_backend(self, messages: List[Dict], temperature: float, stream: bool = False):
        """Call the configured backend."""
        if self.config.BACKEND_TYPE == "ollama":
            return call_ollama(messages, self.tools, self.config, temperature, stream)
        else:  # lmstudio
            return call_lmstudio(messages, self.tools, self.config, temperature, stream)

    def check_backend_health(self) -> bool:
        """Check if the backend is healthy and reachable.

        Returns:
            True if backend is healthy, False otherwise
        """
        if self.config.BACKEND_TYPE == "ollama":
            is_healthy, message = check_ollama_health(self.config, timeout=self.config.HEALTH_CHECK_TIMEOUT)
        else:  # lmstudio
            is_healthy, message = check_lmstudio_health(self.config, timeout=self.config.HEALTH_CHECK_TIMEOUT)

        if is_healthy:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")

        return is_healthy

    def process_chat_completion(self, messages: List[Dict], temperature: float, max_iterations: int = 5) -> Dict:
        """Process chat completion with tool calling loop (non-streaming)."""
        # Add system prompt
        system_prompt = self.get_system_prompt()
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Call the backend with timeout handling
            try:
                response = self.call_backend(full_messages, temperature, stream=False)
                response_data = response.json()
            except requests.Timeout:
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Error: Backend request timed out after {self.config.BACKEND_READ_TIMEOUT}s. The model may be overloaded or unresponsive.",
                            },
                            "finish_reason": "error",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            except requests.ConnectionError:
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Error: Could not connect to {self.config.BACKEND_TYPE} backend at {self.config.LMSTUDIO_ENDPOINT if self.config.BACKEND_TYPE == 'lmstudio' else self.config.OLLAMA_ENDPOINT}. Please ensure the backend is running.",
                            },
                            "finish_reason": "error",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

            # Handle Ollama response format
            if self.config.BACKEND_TYPE == "ollama":
                message = response_data.get("message", {})
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    return {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": self.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": message.get("content", "")},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }

                full_messages.append(message)
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    tool_result = self.execute_tool(function.get("name"), function.get("arguments", {}))
                    full_messages.append({"role": "tool", "content": tool_result})

            else:  # LM Studio (OpenAI format)
                choice = response_data.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    response_data["model"] = self.model_name
                    return response_data

                full_messages.append(message)
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    tool_args = json.loads(function.get("arguments", "{}"))
                    tool_result = self.execute_tool(function.get("name"), tool_args)
                    full_messages.append({"role": "tool", "tool_call_id": tool_call.get("id"), "content": tool_result})

        # Max iterations reached
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I apologize, but I've reached the maximum number of tool calling iterations.",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def stream_chat_response(
        self, messages: List[Dict], temperature: float, max_iterations: int = 5
    ) -> Generator[str, None, None]:
        """Stream chat completion with tool calling loop."""
        system_prompt = self.get_system_prompt()
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Call backend with timeout handling
            try:
                response = self.call_backend(full_messages, temperature, stream=False)
                response_data = response.json()
            except requests.Timeout:
                error_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"Error: Backend request timed out after {self.config.BACKEND_READ_TIMEOUT}s. The model may be overloaded or unresponsive."
                            },
                            "finish_reason": "error",
                        }
                    ],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
            except requests.ConnectionError:
                error_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"Error: Could not connect to {self.config.BACKEND_TYPE} backend. Please ensure it is running."
                            },
                            "finish_reason": "error",
                        }
                    ],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Extract message and tool calls based on backend
            if self.config.BACKEND_TYPE == "ollama":
                message = response_data.get("message", {})
                tool_calls = message.get("tool_calls", [])
            else:  # LM Studio
                choice = response_data.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # No tool calls, stream the final response
                content = message.get("content", "")
                tokens = re.split(r"(\s+)", content)

                for token in tokens:
                    if token:
                        chunk = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.model_name,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk
                final_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Tool calls present - execute them
            full_messages.append(message)

            # Execute each tool
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name")

                if self.config.BACKEND_TYPE == "lmstudio":
                    tool_args = json.loads(function.get("arguments", "{}"))
                else:
                    tool_args = function.get("arguments", {})

                tool_result = self.execute_tool(tool_name, tool_args)

                # Add tool result
                if self.config.BACKEND_TYPE == "lmstudio":
                    full_messages.append({"role": "tool", "tool_call_id": tool_call.get("id"), "content": tool_result})
                else:
                    full_messages.append({"role": "tool", "content": tool_result})

        # Max iterations reached
        error_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "I apologize, but I've reached the maximum number of tool calling iterations."
                    },
                    "finish_reason": "length",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    def health(self):
        """Health check endpoint."""
        return jsonify({"status": "healthy", "backend": self.config.BACKEND_TYPE, "model": self.model_name})

    def list_models(self):
        """List available models."""
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": self.name.lower(),
                        "permission": [],
                        "root": self.model_name,
                        "parent": None,
                    }
                ],
            }
        )

    def chat_completions(self):
        """Handle chat completion requests."""
        try:
            data = request.get_json()

            # Validate JSON payload
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            # Extract and validate required fields
            messages = data.get("messages")
            if messages is None:
                return jsonify({"error": "Missing required field: 'messages'"}), 400

            if not isinstance(messages, list):
                return jsonify({"error": "Field 'messages' must be an array"}), 400

            if not messages:
                return jsonify({"error": "Field 'messages' cannot be empty"}), 400

            temperature = data.get("temperature", self.config.DEFAULT_TEMPERATURE)
            stream = data.get("stream", False)

            if stream:
                return Response(
                    stream_with_context(self.stream_chat_response(messages, temperature)),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            else:
                result = self.process_chat_completion(messages, temperature)
                result["model"] = self.model_name
                return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def run(self, port: Optional[int] = None, host: Optional[str] = None, debug: bool = False, start_webui: bool = True):
        """Run the Flask server.

        Args:
            port: Port to run on (defaults to config.DEFAULT_PORT)
            host: Host to bind to (defaults to config.DEFAULT_HOST, which is 127.0.0.1 for security)
            debug: Enable debug mode
            start_webui: Whether to start Open Web UI
        """
        port = port or self.config.DEFAULT_PORT
        host = host or self.config.DEFAULT_HOST

        print(
            f"""
╭────────────────────────────────────╮
│  {self.name} - AI Assistant with Tools   │
╰────────────────────────────────────╯

Backend: {self.config.BACKEND_TYPE}
Model: {self.config.BACKEND_MODEL}
Host: {host}
Port: {port}
API: http://localhost:{port}/v1
"""
        )

        # Security warning if binding to all interfaces
        if host == "0.0.0.0":
            print("⚠️  WARNING: Server is binding to 0.0.0.0 (all network interfaces)")
            print("   This exposes the API to your entire network without authentication.")
            print("   For security, use HOST=127.0.0.1 (localhost only) unless you need network access.\n")

        # Check backend health if enabled
        if self.config.HEALTH_CHECK_ON_STARTUP:
            print("Checking backend health...")
            if not self.check_backend_health():
                print("\n⚠️  Warning: Backend health check failed!")
                print("The server will start anyway, but requests may fail.")
                print("To disable this check, set HEALTH_CHECK_ON_STARTUP=false\n")

        # Run initialization hook if provided
        if self.init_hook:
            try:
                self.init_hook()
            except Exception as e:
                print(f"Warning: Initialization hook failed: {e}")

        # Start WebUI if requested
        if start_webui and self.config.ENABLE_WEBUI:
            from .webui import start_webui as start_webui_func

            self._webui_process = start_webui_func(port, self.model_name, self.config)

        # Start Flask app
        self.app.run(host=host, port=port, debug=debug)
