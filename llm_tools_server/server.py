"""Core LLM API Server implementation."""

import contextlib
import json
import logging
import re
import threading
import time
import traceback
from collections.abc import Callable, Generator
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from langchain_core.tools import BaseTool

from .backends import call_lmstudio, call_ollama, check_lmstudio_health, check_ollama_health
from .config import ServerConfig


class LLMServer:
    """Flask server providing OpenAI-compatible API for LLM backends with tool calling."""

    def __init__(
        self,
        name: str,
        model_name: str,
        tools: list[BaseTool],
        config: ServerConfig,
        default_system_prompt: str = "You are a helpful AI assistant.",
        init_hook: Callable[[], None] | None = None,
        logger_names: list[str] | None = None,
        rag_index: Any | None = None,
    ):
        """Initialize LLM API Server.

        Args:
            name: Display name for the server (e.g., "MyApp", "DocBot")
            model_name: Model identifier to advertise (e.g., "myapp/assistant")
            tools: List of LangChain tools
            config: ServerConfig instance
            default_system_prompt: Default system prompt if file doesn't exist
            init_hook: Optional function to call during initialization (e.g., index building)
            logger_names: Optional list of logger names for debug logging
            rag_index: Optional DocSearchIndex for pausing background processing during requests
        """
        self.name = name
        self.model_name = model_name
        self.tools = tools
        self.config = config
        self.default_system_prompt = default_system_prompt
        self.init_hook = init_hook
        self.rag_index = rag_index

        # System prompt caching with thread safety
        self._system_prompt_cache: str | None = None
        self._system_prompt_mtime: float | None = None
        self._prompt_lock = threading.Lock()

        # WebUI process
        self._webui_process = None

        # Rate limiter (initialized if enabled)
        self._limiter = None

        # Create Flask app with unique name to avoid conflicts
        # Use full name with prefix to ensure uniqueness across projects
        flask_app_name = f"llm_tools_server_{name.lower().replace(' ', '_')}"
        self.app = Flask(flask_app_name)
        CORS(self.app)

        # Configure rate limiting if enabled
        if config.RATE_LIMIT_ENABLED:
            try:
                from flask_limiter import Limiter
                from flask_limiter.util import get_remote_address

                self._limiter = Limiter(
                    get_remote_address,
                    app=self.app,
                    default_limits=[config.RATE_LIMIT_DEFAULT],
                    storage_uri=config.RATE_LIMIT_STORAGE_URI,
                )
                print(f"Rate limiting enabled: {config.RATE_LIMIT_DEFAULT}")
            except ImportError:
                print("Warning: RATE_LIMIT_ENABLED=true but flask-limiter not installed")
                print("  Install with: pip install flask-limiter")

        # Configure logging
        # Generate default logger name from app name (sanitized: lowercase, spaces to underscores)
        default_logger_name = f"{name.lower().replace(' ', '_')}.tools"
        logger_names = logger_names or [default_logger_name, "tools"]
        # Use the first logger name as the primary logger
        self.logger = logging.getLogger(logger_names[0])

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

            # For JSON format, use minimal formatter (just the message) for pure JSON lines
            # that can be parsed with tools like jq. For text/yaml, include metadata prefix.
            if config.DEBUG_LOG_FORMAT == "json":
                formatter = logging.Formatter("%(message)s")
            else:
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
            print(f"  Format: {config.DEBUG_LOG_FORMAT}")
            print(f"  Logging: {', '.join(logger_names)}")
            print(f"  Rotation: {max_mb:.1f}MB max, {config.DEBUG_LOG_BACKUP_COUNT} backups")
        else:
            # Enable info-level logging to console for request/response visibility
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
                self.logger.addHandler(console_handler)

        # Register routes
        self._register_routes()

    def _log_event(
        self,
        level: str,
        event: str,
        message: str,
        **extra_fields: Any,
    ):
        """Log an event in the configured format (text, json, or yaml).

        This is the central logging method that respects DEBUG_LOG_FORMAT.
        All logging should go through this method to ensure consistent formatting.

        Args:
            level: Log level ("debug", "info", "warning", "error")
            event: Event type (e.g., "request_started", "tool_call", "backend_call")
            message: Human-readable message for text format
            **extra_fields: Additional fields to include in structured formats
        """
        log_format = self.config.DEBUG_LOG_FORMAT
        log_func = getattr(self.logger, level.lower(), self.logger.info)

        if log_format == "json":
            import datetime

            log_entry = {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "level": level.upper(),
                "event": event,
                "message": message,
                **extra_fields,
            }
            log_func(json.dumps(log_entry, ensure_ascii=False, default=str))

        elif log_format == "yaml":
            import datetime

            timestamp = datetime.datetime.now(datetime.UTC).isoformat()
            lines = [
                "---",
                f"timestamp: {timestamp}",
                f"level: {level.upper()}",
                f"event: {event}",
                f"message: {json.dumps(message)}",
            ]
            for key, value in extra_fields.items():
                if isinstance(value, str) and "\n" in value:
                    lines.append(f"{key}: |")
                    for line in value.split("\n"):
                        lines.append(f"  {line}")
                else:
                    lines.append(f"{key}: {json.dumps(value, default=str)}")
            log_func("\n".join(lines))

        else:  # text format (default)
            log_func(message)

    def _log_tool_event(
        self,
        event: str,
        tool_name: str,
        tool_input: dict[str, Any],
        response: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ):
        """Log a tool event in the configured format (text, json, or yaml).

        Args:
            event: Event type (e.g., "tool_call", "tool_error")
            tool_name: Name of the tool
            tool_input: Input parameters to the tool
            response: Tool response (optional)
            duration_ms: Duration in milliseconds (optional)
            error: Error message if applicable (optional)
        """
        log_format = self.config.DEBUG_LOG_FORMAT
        max_len = self.config.DEBUG_LOG_MAX_RESPONSE_LENGTH

        # Apply truncation if configured
        log_response = response
        if response and max_len > 0 and len(response) > max_len:
            log_response = response[:max_len] + f"\n... (truncated, total: {len(response)} chars)"

        if log_format == "json":
            import datetime

            log_entry = {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "event": event,
                "tool": tool_name,
                "input": tool_input,
            }
            if duration_ms is not None:
                log_entry["duration_ms"] = round(duration_ms, 2)
            if response is not None:
                log_entry["response_length"] = len(response)
                log_entry["response"] = log_response
            if error is not None:
                log_entry["error"] = error
            self.logger.debug(json.dumps(log_entry, ensure_ascii=False))

        elif log_format == "yaml":
            import datetime

            # Build YAML manually to avoid adding pyyaml dependency
            timestamp = datetime.datetime.now(datetime.UTC).isoformat()
            lines = [
                "---",
                f"timestamp: {timestamp}",
                f"event: {event}",
                f"tool: {tool_name}",
                "input:",
            ]
            for key, value in tool_input.items():
                # Simple YAML serialization for common types
                if isinstance(value, str):
                    lines.append(f"  {key}: {json.dumps(value)}")
                else:
                    lines.append(f"  {key}: {json.dumps(value)}")
            if duration_ms is not None:
                lines.append(f"duration_ms: {round(duration_ms, 2)}")
            if response is not None:
                lines.append(f"response_length: {len(response)}")
                # Use YAML literal block for multiline responses
                if "\n" in (log_response or ""):
                    lines.append("response: |")
                    for line in (log_response or "").split("\n"):
                        lines.append(f"  {line}")
                else:
                    lines.append(f"response: {json.dumps(log_response)}")
            if error is not None:
                lines.append(f"error: {json.dumps(error)}")
            self.logger.debug("\n".join(lines))

        else:  # text format (default)
            self.logger.debug("=" * 80)
            self.logger.debug(f"TOOL CALL: {tool_name}")
            self.logger.debug(f"INPUT: {json.dumps(tool_input, indent=2)}")
            if duration_ms is not None:
                self.logger.debug(f"DURATION: {round(duration_ms, 2)}ms")
            if error is not None:
                self.logger.debug(f"ERROR: {error}")
            elif response is not None:
                self.logger.debug(f"RESPONSE: {log_response}")
            self.logger.debug("=" * 80 + "\n")

    def _register_routes(self):
        """Register Flask routes."""
        self.app.route("/health", methods=["GET"])(self.health)
        self.app.route("/v1/models", methods=["GET"])(self.list_models)
        self.app.route("/v1/chat/completions", methods=["POST"])(self.chat_completions)
        self.app.route("/config/model", methods=["GET", "POST"])(self.config_model)

    def get_system_prompt(self) -> str:
        """Load system prompt from markdown file with smart caching.

        Thread-safe: uses lock to prevent race conditions when multiple
        requests check and update the cache simultaneously.
        """
        prompt_path = Path(self.config.SYSTEM_PROMPT_PATH)

        if not prompt_path.exists():
            return self.default_system_prompt

        try:
            with self._prompt_lock:
                current_mtime = prompt_path.stat().st_mtime

                # Check if cache is valid
                if self._system_prompt_cache is not None and self._system_prompt_mtime == current_mtime:
                    return self._system_prompt_cache

                # Read and cache the prompt
                content = prompt_path.read_text(encoding="utf-8")

                # Verify mtime didn't change during read (file was modified)
                if prompt_path.stat().st_mtime != current_mtime:
                    # File changed during read, re-read to get consistent content
                    content = prompt_path.read_text(encoding="utf-8")
                    current_mtime = prompt_path.stat().st_mtime

                self._system_prompt_cache = content
                self._system_prompt_mtime = current_mtime

                return self._system_prompt_cache
        except Exception as e:
            print(f"Error reading system prompt: {e}")
            return self.default_system_prompt

    def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool by name with given input."""
        start_time = time.time()

        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.func(**tool_input)
                    result_str = str(result)
                    duration_ms = (time.time() - start_time) * 1000

                    # Log tool call with response
                    if self.config.DEBUG_TOOLS:
                        self._log_tool_event(
                            event="tool_call",
                            tool_name=tool_name,
                            tool_input=tool_input,
                            response=result_str,
                            duration_ms=duration_ms,
                        )

                    return result_str
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    error_msg = f"Error executing tool '{tool_name}': {type(e).__name__}: {e!s}"
                    full_error = f"{error_msg}\n\nFull traceback:\n{traceback.format_exc()}"

                    # Log tool error
                    if self.config.DEBUG_TOOLS:
                        self._log_tool_event(
                            event="tool_error",
                            tool_name=tool_name,
                            tool_input=tool_input,
                            duration_ms=duration_ms,
                            error=full_error,
                        )

                    # Return user-friendly error message
                    return f"{error_msg}\n\nTip: Enable DEBUG_TOOLS for detailed error logs."

        # Tool not found
        duration_ms = (time.time() - start_time) * 1000
        available_tools = [t.name for t in self.tools]
        not_found_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"

        if self.config.DEBUG_TOOLS:
            self._log_tool_event(
                event="tool_not_found",
                tool_name=tool_name,
                tool_input=tool_input,
                duration_ms=duration_ms,
                error=not_found_msg,
            )

        return not_found_msg

    def call_backend(
        self, messages: list[dict], temperature: float, stream: bool = False, tool_choice: str | None = None
    ):
        """Call the configured backend.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            stream: Whether to stream the response
            tool_choice: Tool calling mode - "required", "auto", or "none"
        """
        start_time = time.time()
        if self.config.BACKEND_TYPE == "ollama":
            result = call_ollama(messages, self.tools, self.config, temperature, stream, tool_choice)
        else:  # lmstudio
            result = call_lmstudio(messages, self.tools, self.config, temperature, stream, tool_choice)
        if not stream:
            duration = time.time() - start_time
            self._log_event(
                "info",
                "backend_call",
                f"Backend call: {self.config.BACKEND_TYPE}/{self.config.BACKEND_MODEL} in {duration:.2f}s",
                backend=self.config.BACKEND_TYPE,
                model=self.config.BACKEND_MODEL,
                duration_s=round(duration, 2),
            )
        return result

    def _extract_message_and_tool_calls(self, response_data: dict) -> tuple[dict, list]:
        """Extract message and tool_calls from backend response.

        Args:
            response_data: JSON response from backend

        Returns:
            Tuple of (message dict, tool_calls list)
        """
        if self.config.BACKEND_TYPE == "ollama":
            message = response_data.get("message", {})
            tool_calls = message.get("tool_calls", [])
        else:  # LM Studio (OpenAI format)
            choice = response_data.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

        # Handle thinker models (like Apriel) that embed responses in markers
        if not tool_calls:
            content = message.get("content", "")
            if "[BEGIN FINAL RESPONSE]" in content:
                cleaned_content, tool_calls = self._parse_thinker_response(content)
                # Update message with cleaned content (without reasoning)
                message = dict(message)
                message["content"] = cleaned_content

        return message, tool_calls

    def _contains_malformed_tool_tokens(self, content: str) -> bool:
        """Check if response content contains malformed tool call tokens.

        Some models output raw internal function calling tokens like:
        <|start|>assistant<|channel|>commentary to=functions.web_search <|constrain|>json<|message|>{...}

        These should have been parsed as tool calls but weren't.

        Args:
            content: Response content to check

        Returns:
            True if malformed tokens are detected
        """
        if not content:
            return False

        # Pattern for Hermes/ChatML-style malformed tool call tokens
        patterns = [
            # Hermes-style: <|start|>assistant<|channel|>...
            r"<\|start\|>assistant<\|channel\|>",
            # Generic special token patterns that indicate malformed output
            r"<\|start\|>.*?<\|message\|>",
            # Functions marker
            r"to=functions\.\w+",
        ]

        return any(re.search(pattern, content, re.DOTALL) for pattern in patterns)

    def _parse_thinker_response(self, content: str) -> tuple[str, list]:
        """Parse response from thinker models that include reasoning.

        Extracts content from [BEGIN FINAL RESPONSE]...[END FINAL RESPONSE] markers
        and parses any embedded <tool_calls> within.

        Args:
            content: Full message content including reasoning

        Returns:
            Tuple of (cleaned content, tool_calls list)
        """
        # Extract content between markers
        match = re.search(
            r"\[BEGIN FINAL RESPONSE\]\s*(.*?)\s*\[END FINAL RESPONSE\]",
            content,
            re.DOTALL,
        )
        if not match:
            return content, []

        final_content = match.group(1).strip()

        # Check for embedded tool calls within the final response
        tool_calls = []
        if "<tool_calls>" in final_content:
            tool_match = re.search(r"<tool_calls>\s*(\[.*?\])\s*</tool_calls>", final_content, re.DOTALL)
            if tool_match:
                try:
                    raw_calls = json.loads(tool_match.group(1))
                    for i, call in enumerate(raw_calls):
                        tool_calls.append(
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": call.get("name"),
                                    "arguments": json.dumps(call.get("arguments", {})),
                                },
                            }
                        )
                    self._log_event(
                        "debug",
                        "thinker_tool_calls_parsed",
                        f"Parsed {len(tool_calls)} embedded tool calls",
                        tool_call_count=len(tool_calls),
                    )
                    # Remove tool_calls tag from content since we're returning them separately
                    final_content = re.sub(r"<tool_calls>.*?</tool_calls>", "", final_content, flags=re.DOTALL).strip()
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    self._log_event(
                        "warning",
                        "thinker_tool_calls_parse_error",
                        f"Failed to parse embedded tool calls: {e}",
                        error=str(e),
                    )

        return final_content, tool_calls

    def _execute_tool_calls(self, tool_calls: list, tools_used: list[str]) -> list[dict]:
        """Execute tool calls and return formatted result messages.

        Args:
            tool_calls: List of tool call objects from backend
            tools_used: List to append tool names to (modified in place)

        Returns:
            List of tool result messages formatted for the backend
        """
        result_messages = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            if tool_name:
                tools_used.append(tool_name)

            # Parse arguments (LM Studio sends JSON string, Ollama sends dict)
            if self.config.BACKEND_TYPE == "lmstudio":
                tool_args = json.loads(function.get("arguments", "{}"))
            else:
                tool_args = function.get("arguments", {})

            tool_result = self.execute_tool(tool_name, tool_args)

            # Format result message for backend
            if self.config.BACKEND_TYPE == "lmstudio":
                result_messages.append({"role": "tool", "tool_call_id": tool_call.get("id"), "content": tool_result})
            else:
                result_messages.append({"role": "tool", "content": tool_result})

        return result_messages

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

    def process_chat_completion(
        self, messages: list[dict], temperature: float, max_iterations: int | None = None
    ) -> dict:
        """Process chat completion with tool calling loop (non-streaming)."""
        if max_iterations is None:
            max_iterations = self.config.MAX_TOOL_ITERATIONS

        # Reset retry flag for tool_choice=required nudge (only retry once per request)
        self._tool_required_retry_done = False

        # Add system prompt
        system_prompt = self.get_system_prompt()
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        # Track tools used during this request
        tools_used: list[str] = []

        # Track tool loop timing
        tool_loop_start = time.time()
        timeout = self.config.TOOL_LOOP_TIMEOUT

        iteration = 0
        while iteration < max_iterations:
            # Check timeout (if enabled)
            if timeout > 0 and (time.time() - tool_loop_start) > timeout:
                self._log_event(
                    "warning",
                    "tool_loop_timeout",
                    f"[TOOL LOOP] TIMEOUT after {timeout}s. Tools used: {tools_used}",
                    timeout_s=timeout,
                    tools_used=tools_used,
                )
                break
            iteration += 1
            # Determine tool_choice based on iteration:
            # - First iteration: configurable (default "auto")
            # - Subsequent iterations: always "auto" to let model decide
            tool_choice = self.config.FIRST_ITERATION_TOOL_CHOICE if iteration == 1 else "auto"

            self._log_event(
                "debug",
                "tool_loop_iteration",
                f"[TOOL LOOP] Iteration {iteration}/{max_iterations} (tool_choice={tool_choice})",
                iteration=iteration,
                max_iterations=max_iterations,
                tool_choice=tool_choice,
            )

            # Call the backend with timeout handling
            try:
                response = self.call_backend(full_messages, temperature, stream=False, tool_choice=tool_choice)
                response_data = response.json()
            except requests.Timeout:
                backend_endpoint = (
                    self.config.LMSTUDIO_ENDPOINT
                    if self.config.BACKEND_TYPE == "lmstudio"
                    else self.config.OLLAMA_ENDPOINT
                )
                error_content = (
                    f"Backend request timed out after {self.config.BACKEND_READ_TIMEOUT}s.\n\n"
                    f"Backend: {self.config.BACKEND_TYPE}\n"
                    f"Endpoint: {backend_endpoint}\n"
                    f"Model: {self.config.BACKEND_MODEL}\n\n"
                    f"Troubleshooting:\n"
                    f"• The model may be overloaded or generating a very long response\n"
                    f"• Increase timeout: Set BACKEND_READ_TIMEOUT={self.config.BACKEND_READ_TIMEOUT * 2}\n"
                    f"• Check backend logs for errors\n"
                    f"• Try a smaller/faster model"
                )
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": error_content},
                            "finish_reason": "error",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            except requests.ConnectionError as e:
                backend_endpoint = (
                    self.config.LMSTUDIO_ENDPOINT
                    if self.config.BACKEND_TYPE == "lmstudio"
                    else self.config.OLLAMA_ENDPOINT
                )
                error_content = (
                    f"Could not connect to {self.config.BACKEND_TYPE} backend.\n\n"
                    f"Backend: {self.config.BACKEND_TYPE}\n"
                    f"Endpoint: {backend_endpoint}\n"
                    f"Model: {self.config.BACKEND_MODEL}\n"
                    f"Error: {type(e).__name__}: {e!s}\n\n"
                    f"Troubleshooting:\n"
                    f"• Ensure {self.config.BACKEND_TYPE} is running\n"
                    f"• Verify endpoint URL is correct\n"
                    f"• Check if model is loaded in backend\n"
                    f"• Test connection: curl {backend_endpoint}/models"
                )
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": error_content},
                            "finish_reason": "error",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

            # Extract message and tool calls from response
            message, tool_calls = self._extract_message_and_tool_calls(response_data)

            # Retry once if tool_choice was "required" but model didn't call any tools
            if not tool_calls and tool_choice == "required" and not getattr(self, "_tool_required_retry_done", False):
                self._tool_required_retry_done = True
                self._log_event(
                    "warning",
                    "tool_choice_required_ignored",
                    "[TOOL LOOP] tool_choice=required was ignored by model, retrying with nudge",
                    iteration=iteration,
                )
                # Add a nudge message to encourage tool use
                nudge_message = {
                    "role": "user",
                    "content": "Please use one of the available tools to help answer this question.",
                }
                full_messages.append(nudge_message)
                continue  # Retry this iteration

            if not tool_calls:
                # No tool calls - return final response using the cleaned message
                # (which has thinker model reasoning stripped if applicable)
                content = message.get("content", "")

                # Log warning if response is empty after tool calls were made
                if not content and tools_used:
                    self._log_event(
                        "warning",
                        "empty_response_after_tools",
                        f"Empty response from model after {len(tools_used)} tool calls: {tools_used}",
                        tools_used=tools_used,
                        iteration=iteration,
                    )

                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "tools_used": tools_used,
                }

            # Execute tool calls and append results
            # Only include standard message fields to avoid confusing the model
            # (some models add non-standard fields like 'reasoning' that shouldn't be sent back)
            clean_message = {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            }
            # Include tool_calls if present (required for proper conversation flow)
            if "tool_calls" in message:
                clean_message["tool_calls"] = message["tool_calls"]
            full_messages.append(clean_message)
            tool_results = self._execute_tool_calls(tool_calls, tools_used)
            full_messages.extend(tool_results)

        # Tool loop limit reached (timeout or max iterations) - force a final response
        # Log which limit was hit (timeout was logged above via break, max iterations below)
        if iteration >= max_iterations:
            self._log_event(
                "warning",
                "tool_loop_max_iterations",
                f"[TOOL LOOP] MAX ITERATIONS REACHED ({max_iterations}). Tools used: {tools_used}",
                max_iterations=max_iterations,
                tools_used=tools_used,
            )

        # Generate final response by calling backend WITHOUT tools
        # This forces the LLM to produce a text response rather than more tool calls
        return self._generate_final_response(full_messages, temperature, tools_used)

    def _generate_final_response(self, messages: list[dict], temperature: float, tools_used: list[str]) -> dict:
        """Generate a final response without tools after hitting loop limits.

        When the tool loop reaches max iterations or timeout, we call the backend
        one more time WITHOUT tools. This forces the LLM to produce a text response
        synthesizing the information gathered rather than attempting more tool calls.

        If the model outputs malformed tool tokens instead of a proper response,
        we retry once with a stern message instructing it to just answer.

        Args:
            messages: Full message history including tool results
            temperature: Sampling temperature
            tools_used: List of tool names used during the request
        """
        self._log_event(
            "info",
            "tool_loop_final_response",
            "[TOOL LOOP] Generating final response without tools (tool_choice=none)",
        )

        # Log full message history if debug tools is enabled
        if self.config.DEBUG_TOOLS:
            self._log_event(
                "debug",
                "tool_loop_messages_payload",
                f"[TOOL LOOP] Final response messages payload ({len(messages)} messages)",
                message_count=len(messages),
                messages=messages,
            )

        # Try up to 2 times - first normally, then with stern instruction if malformed
        max_attempts = 2
        current_messages = messages.copy()

        for attempt in range(max_attempts):
            try:
                if self.config.BACKEND_TYPE == "ollama":
                    from .backends import call_ollama

                    response = call_ollama(
                        current_messages,
                        [],  # Empty tools list
                        self.config,
                        temperature,
                        stream=False,
                        tool_choice="none",  # Explicitly disable tool calls
                    )
                else:  # lmstudio
                    from .backends import call_lmstudio

                    response = call_lmstudio(
                        current_messages,
                        [],  # Empty tools list
                        self.config,
                        temperature,
                        stream=False,
                        tool_choice="none",  # Explicitly disable tool calls
                    )
                response_data = response.json()
            except (requests.Timeout, requests.ConnectionError) as e:
                self._log_event(
                    "error",
                    "tool_loop_final_response_error",
                    f"[TOOL LOOP] Failed to generate final response: {e}",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return self._make_error_response(
                    "I gathered some information but encountered an error generating a response. Please try again.",
                    tools_used,
                )

            # Extract message
            message, _ = self._extract_message_and_tool_calls(response_data)
            content = message.get("content", "")

            # Check for malformed tool tokens or empty response
            has_malformed = self._contains_malformed_tool_tokens(content)
            is_empty = not content or not content.strip()

            if has_malformed or is_empty:
                if attempt == 0:
                    # First attempt failed - retry with stern instruction
                    self._log_event(
                        "warning",
                        "malformed_response_retry",
                        "[TOOL LOOP] Response contains malformed tokens or is empty, retrying with stern instruction",
                        has_malformed=has_malformed,
                        is_empty=is_empty,
                        raw_content=content[:200] if content else "(empty)",
                    )
                    # Add stern instruction to force a proper response
                    stern_message = {
                        "role": "user",
                        "content": (
                            "IMPORTANT: Do not attempt to use any tools. Tools are not available. "
                            "Based on the information you have gathered, provide your best answer now. "
                            "Just give a direct response to the original question."
                        ),
                    }
                    current_messages = messages.copy()
                    current_messages.append(stern_message)
                    continue
                else:
                    # Second attempt also failed - give up and return fallback
                    self._log_event(
                        "error",
                        "malformed_response_fallback",
                        "[TOOL LOOP] Response still malformed after retry, returning fallback",
                        has_malformed=has_malformed,
                        is_empty=is_empty,
                        tools_used=tools_used,
                    )
                    return self._make_error_response(
                        "I'm sorry, I encountered an issue processing your request. Please try again.",
                        tools_used,
                    )
            else:
                # Valid response - return it
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "tools_used": tools_used,
                }

        # Should not reach here, but just in case
        return self._make_error_response(
            "I'm sorry, I encountered an issue processing your request. Please try again.",
            tools_used,
        )

    def _make_error_response(self, message: str, tools_used: list[str]) -> dict:
        """Create a standardized error response."""
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message},
                    "finish_reason": "error",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "tools_used": tools_used,
        }

    def _stream_from_backend(self, messages: list[dict], temperature: float) -> Generator[str, None, None]:
        """Stream response directly from backend, converting to OpenAI SSE format.

        This performs true streaming - tokens are sent as they're generated by the backend.
        For thinker models, buffers content and filters out reasoning, only streaming
        the content between [BEGIN FINAL RESPONSE] and [END FINAL RESPONSE] markers.
        """
        chat_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        # Buffer for thinker model detection and filtering
        content_buffer = ""
        full_content = ""  # Keep all content in case model doesn't use markers
        in_final_response = False
        found_markers = False  # Track if we ever found thinker markers
        begin_marker = "[BEGIN FINAL RESPONSE]"
        end_marker = "[END FINAL RESPONSE]"

        def make_chunk(content: str) -> str:
            """Create an SSE chunk with content."""
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        def process_buffered_content() -> Generator[str, None, None]:
            """Process buffered content for thinker model markers."""
            nonlocal content_buffer, in_final_response, found_markers

            while True:
                if not in_final_response:
                    # Look for begin marker
                    begin_idx = content_buffer.find(begin_marker)
                    if begin_idx != -1:
                        # Found begin marker - discard everything before it (reasoning)
                        found_markers = True
                        content_buffer = content_buffer[begin_idx + len(begin_marker) :]
                        in_final_response = True
                        # Strip leading whitespace after marker
                        content_buffer = content_buffer.lstrip()
                    else:
                        # No begin marker yet - keep buffer for potential marker detection
                        # Don't discard anything - we'll decide at the end if model uses markers
                        break
                else:
                    # In final response - look for end marker
                    end_idx = content_buffer.find(end_marker)
                    if end_idx != -1:
                        # Found end marker - yield content before it, discard rest
                        final_content = content_buffer[:end_idx].rstrip()
                        if final_content:
                            yield make_chunk(final_content)
                        content_buffer = ""
                        in_final_response = False
                        break
                    else:
                        # No end marker yet - yield safe content
                        # Keep last len(end_marker)-1 chars in case marker spans chunks
                        safe_len = len(content_buffer) - (len(end_marker) - 1)
                        if safe_len > 0:
                            yield make_chunk(content_buffer[:safe_len])
                            content_buffer = content_buffer[safe_len:]
                        break

        try:
            response = self.call_backend(messages, temperature, stream=True)

            if self.config.BACKEND_TYPE == "ollama":
                # Ollama streams newline-delimited JSON
                for line in response.iter_lines():
                    if line:
                        chunk_data = json.loads(line)
                        message = chunk_data.get("message", {})
                        content = message.get("content", "")
                        done = chunk_data.get("done", False)

                        if content:
                            full_content += content  # Always track full content
                            content_buffer += content
                            yield from process_buffered_content()

                        if done:
                            break
            else:
                # LM Studio uses OpenAI SSE format (data: {...}\n\n)
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove "data: " prefix
                            if data_str == "[DONE]":
                                break
                            chunk_data = json.loads(data_str)
                            choice = chunk_data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                full_content += content  # Always track full content
                                content_buffer += content
                                yield from process_buffered_content()

            # Flush any remaining buffered content
            if found_markers:
                # Model uses thinker markers - flush any remaining content from marker processing
                if content_buffer:
                    if in_final_response:
                        # Remove any trailing end marker if present
                        if content_buffer.endswith(end_marker):
                            content_buffer = content_buffer[: -len(end_marker)]
                        content_buffer = content_buffer.rstrip()
                    if content_buffer:
                        yield make_chunk(content_buffer)
            elif full_content:
                # Model doesn't use thinker markers - output all accumulated content
                yield make_chunk(full_content)

            # Final chunk with finish_reason
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except requests.Timeout:
            yield from self._yield_error_chunk(
                f"Backend request timed out after {self.config.BACKEND_READ_TIMEOUT}s.\n\n"
                f"Backend: {self.config.BACKEND_TYPE}\n"
                f"Try: Increase BACKEND_READ_TIMEOUT or use a faster model"
            )
        except requests.ConnectionError:
            yield from self._yield_error_chunk(
                f"Could not connect to {self.config.BACKEND_TYPE} backend.\n\n"
                f"Ensure {self.config.BACKEND_TYPE} is running and accessible"
            )

    def _yield_error_chunk(self, error_content: str) -> Generator[str, None, None]:
        """Yield an error as an SSE chunk."""
        error_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {"content": error_content}, "finish_reason": "error"}],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    def stream_chat_response(
        self, messages: list[dict], temperature: float, max_iterations: int | None = None
    ) -> Generator[str, None, None]:
        """Stream chat completion with tool calling loop.

        Tool-calling iterations use non-streaming requests to detect tool calls.
        The final response (no tool calls) uses true streaming from the backend.
        """
        if max_iterations is None:
            max_iterations = self.config.MAX_TOOL_ITERATIONS

        system_prompt = self.get_system_prompt()
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            self._log_event(
                "debug",
                "tool_loop_iteration",
                f"[TOOL LOOP] Iteration {iteration}/{max_iterations}",
                iteration=iteration,
                max_iterations=max_iterations,
                streaming=True,
            )

            # Call backend without streaming to check for tool calls
            try:
                response = self.call_backend(full_messages, temperature, stream=False)
                response_data = response.json()
            except requests.Timeout:
                yield from self._yield_error_chunk(
                    f"Backend request timed out after {self.config.BACKEND_READ_TIMEOUT}s.\n\n"
                    f"Backend: {self.config.BACKEND_TYPE}\n"
                    f"Try: Increase BACKEND_READ_TIMEOUT or use a faster model"
                )
                return
            except requests.ConnectionError:
                yield from self._yield_error_chunk(
                    f"Could not connect to {self.config.BACKEND_TYPE} backend.\n\n"
                    f"Ensure {self.config.BACKEND_TYPE} is running and accessible"
                )
                return

            # Extract message and tool calls from response
            message, tool_calls = self._extract_message_and_tool_calls(response_data)

            if not tool_calls:
                # No tool calls - use true streaming for final response
                yield from self._stream_from_backend(full_messages, temperature)
                return

            # Execute tool calls and append results (tools_used not tracked in streaming)
            full_messages.append(message)
            tool_results = self._execute_tool_calls(tool_calls, [])
            full_messages.extend(tool_results)

        # Max iterations reached
        self._log_event(
            "warning",
            "tool_loop_max_iterations",
            f"[TOOL LOOP] MAX ITERATIONS REACHED ({max_iterations}) in streaming mode",
            max_iterations=max_iterations,
            streaming=True,
        )
        yield from self._yield_error_chunk(
            "I apologize, but I've reached the maximum number of tool calling iterations."
        )

    def health(self):
        """Health check endpoint."""
        return jsonify({"status": "healthy", "backend": self.config.BACKEND_TYPE, "model": self.model_name})

    def config_model(self):
        """Get or set the backend model.

        GET: Returns current backend model configuration
        POST: Sets a new backend model (no restart required)

        POST body: {"model": "model-name"}
        """
        if request.method == "GET":
            return jsonify(
                {
                    "backend_model": self.config.BACKEND_MODEL,
                    "backend_type": self.config.BACKEND_TYPE,
                }
            )

        # POST - set new model
        data = request.get_json() or {}
        new_model = data.get("model")

        if not new_model:
            return jsonify({"error": "Missing 'model' in request body"}), 400

        old_model = self.config.BACKEND_MODEL
        self.config.BACKEND_MODEL = new_model
        self._log_event(
            "info",
            "backend_model_changed",
            f"Backend model changed: {old_model} -> {new_model}",
            previous_model=old_model,
            current_model=new_model,
        )

        return jsonify(
            {
                "status": "ok",
                "previous_model": old_model,
                "current_model": new_model,
            }
        )

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
        start_time = time.time()

        # Pause background RAG processing during user request
        if self.rag_index is not None:
            with contextlib.suppress(Exception):
                self.rag_index.pause_background_processing()

        try:
            data = request.get_json(silent=True)

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

            # Allow request to override backend model (passthrough to LM Studio/Ollama)
            request_model = data.get("model")
            original_model = None
            if request_model and request_model != self.model_name:
                # Temporarily override for this request
                original_model = self.config.BACKEND_MODEL
                self.config.BACKEND_MODEL = request_model

            self._log_event(
                "info",
                "request_started",
                f"Request: {len(messages)} messages, stream={stream}, temp={temperature}, model={self.config.BACKEND_MODEL}",
                message_count=len(messages),
                stream=stream,
                temperature=temperature,
                backend_model=self.config.BACKEND_MODEL,
            )

            try:
                if stream:
                    return Response(
                        stream_with_context(self.stream_chat_response(messages, temperature)),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                else:
                    result = self.process_chat_completion(messages, temperature)
                    result["model"] = self.model_name
                    duration = time.time() - start_time
                    tools_used = result.get("tools_used", [])
                    self._log_event(
                        "info",
                        "request_completed",
                        f"Completed in {duration:.2f}s, tools={tools_used}",
                        duration_s=round(duration, 2),
                        tools_used=tools_used,
                    )
                    return jsonify(result)
            finally:
                # Restore original model if we overrode it
                if original_model is not None:
                    self.config.BACKEND_MODEL = original_model

        except Exception as e:
            error_details = {
                "error": f"{type(e).__name__}: {e!s}",
                "type": type(e).__name__,
                "endpoint": "/v1/chat/completions",
            }

            # Add traceback in debug mode
            if self.config.DEBUG_TOOLS:
                error_details["traceback"] = traceback.format_exc()
                self._log_event(
                    "error",
                    "unhandled_exception",
                    f"Unhandled exception in chat_completions: {type(e).__name__}: {e!s}",
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc(),
                )

            return jsonify(error_details), 500

        finally:
            # Resume background RAG processing after request completes
            if self.rag_index is not None:
                with contextlib.suppress(Exception):
                    self.rag_index.resume_background_processing()

    def run(
        self,
        port: int | None = None,
        host: str | None = None,
        debug: bool = False,
        start_webui: bool = True,
        threaded: bool | None = None,
    ):
        """Run the Flask server.

        Args:
            port: Port to run on (defaults to config.DEFAULT_PORT)
            host: Host to bind to (defaults to config.DEFAULT_HOST, which is 127.0.0.1 for security)
            debug: Enable debug mode
            start_webui: Whether to start Open Web UI
            threaded: Enable threaded mode for concurrent requests (defaults to config.THREADED)
                     Note: For production, use a WSGI server like Gunicorn with workers instead
        """
        port = port or self.config.DEFAULT_PORT
        host = host or self.config.DEFAULT_HOST
        threaded = threaded if threaded is not None else self.config.THREADED

        threading_mode = "enabled" if threaded else "disabled"

        # Build dynamic banner with centered text
        title = f"{self.name} - AI Assistant with Tools"
        padding = 2  # spaces on each side
        inner_width = len(title) + (padding * 2)
        top_border = "╭" + "─" * inner_width + "╮"
        middle_line = "│" + " " * padding + title + " " * padding + "│"
        bottom_border = "╰" + "─" * inner_width + "╯"

        print(
            f"""
{top_border}
{middle_line}
{bottom_border}

Backend: {self.config.BACKEND_TYPE}
Model: {self.config.BACKEND_MODEL}
Host: {host}
Port: {port}
Threading: {threading_mode}
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
        self.app.run(host=host, port=port, debug=debug, threaded=threaded)
