"""Open Web UI integration."""

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Optional


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def find_available_port(starting_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from the given port."""
    for i in range(max_attempts):
        port = starting_port + i
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find available port starting from {starting_port}")


def start_webui(
    backend_port: int, model_name: str, config, prompt_suggestions: Optional[list] = None
) -> Optional[subprocess.Popen]:
    """Start Open Web UI as a subprocess.

    Args:
        backend_port: Port where the LLM API server is running
        model_name: Model name to configure as default
        config: ServerConfig instance
        prompt_suggestions: Optional list of prompt suggestions

    Returns:
        Popen process or None if failed to start
    """
    try:
        # Find open-webui executable (prefer venv, fallback to system)
        # Look in the project directory's venv
        venv_openwebui = Path.cwd() / "venv" / "bin" / "open-webui"
        if venv_openwebui.exists():
            openwebui_cmd = str(venv_openwebui)
        else:
            result = subprocess.run(["which", "open-webui"], capture_output=True, text=True)
            if result.returncode != 0:
                print("Warning: open-webui not found. Install with: pip install open-webui")
                return None
            openwebui_cmd = "open-webui"

        # Find an available port
        webui_port = find_available_port(config.WEBUI_PORT)
        print(f"Starting Open Web UI on port {webui_port}...")

        # Set up environment variables
        env = os.environ.copy()
        env["OPENAI_API_BASE_URLS"] = f"http://localhost:{backend_port}/v1"
        env["OPENAI_API_KEYS"] = "sk-local"  # Dummy key
        env["DEFAULT_MODELS"] = model_name

        # Set custom prompt suggestions if provided
        if prompt_suggestions or config.DEFAULT_PROMPT_SUGGESTIONS:
            suggestions = prompt_suggestions or config.DEFAULT_PROMPT_SUGGESTIONS
            env["DEFAULT_PROMPT_SUGGESTIONS"] = json.dumps(suggestions)

        # Start open-webui
        process = subprocess.Popen(
            [openwebui_cmd, "serve", "--port", str(webui_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        print(f"Open Web UI started at http://localhost:{webui_port}")
        print(f"Backend auto-configured at http://localhost:{backend_port}/v1")

        return process

    except Exception as e:
        print(f"Failed to start Open Web UI: {e}")
        return None
