# Feature Request: Support `think` Parameter for Extended Thinking Models

## Summary

Add support for the `think` parameter in llm-tools-server to enable extended thinking/reasoning capabilities for models that support it (DeepSeek R1, Qwen3, gpt-oss, etc.).

## Background

Ollama and LM Studio both support a `think` parameter that controls whether reasoning models show their thinking process:

- **Ollama**: Native support via `think: true/false` in `/api/chat` requests ([Ollama Thinking Blog](https://ollama.com/blog/thinking))
- **LM Studio/gpt-oss**: Supports intensity levels `'low'`, `'medium'`, `'high'` via the `reasoning` parameter ([LangChain Ollama Reference](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html))

When enabled, the model's internal reasoning is separated from its final output, allowing applications to display or process thinking content differently.

## Proposed Changes

### File: `llm_tools_server/backends.py`

1. Add `think` parameter to `call_ollama()`:

```python
def call_ollama(messages: list[dict], tools: list, config, temperature: float = 0.0, stream: bool = False, think: bool | None = None):
    # ... existing code ...

    payload = {
        "model": config.BACKEND_MODEL,
        "messages": messages,
        "tools": ollama_tools,
        "stream": stream,
        "options": {"temperature": temperature},
    }

    # Add think parameter if specified
    if think is not None:
        payload["think"] = think
```

2. Add `think` parameter to `call_lmstudio()`:

```python
def call_lmstudio(messages: list[dict], tools: list, config, temperature: float = 0.0, stream: bool = False, think: str | None = None):
    # ... existing code ...

    payload = {
        "model": config.BACKEND_MODEL,
        "messages": messages,
        "tools": openai_tools,
        "temperature": temperature,
        "stream": stream,
    }

    # Add reasoning parameter if specified (LM Studio uses 'reasoning' for gpt-oss)
    if think is not None:
        payload["reasoning"] = think  # Accepts 'low', 'medium', 'high'
```

### File: `llm_tools_server/server.py`

3. Extract `think` from request and pass through:

```python
def chat_completions(self):
    # ... existing code ...

    temperature = data.get("temperature", self.config.DEFAULT_TEMPERATURE)
    stream = data.get("stream", False)
    think = data.get("think")  # New: extract think parameter

    # Pass think to call_backend and downstream methods
```

4. Update `call_backend()` method:

```python
def call_backend(self, messages: list[dict], temperature: float, stream: bool = False, think: bool | str | None = None):
    if self.config.BACKEND_TYPE == "ollama":
        result = call_ollama(messages, self.tools, self.config, temperature, stream, think=think)
    else:
        result = call_lmstudio(messages, self.tools, self.config, temperature, stream, think=think)
    return result
```

5. Update `process_chat_completion()` and `stream_chat_response()` signatures to accept and pass `think`.

### File: `llm_tools_server/config.py` (Optional)

6. Add default configuration:

```python
# Extended thinking (for reasoning models like DeepSeek R1, Qwen3)
THINK: bool | str | None = None  # None = use model default, True/False for Ollama, 'low'/'medium'/'high' for LM Studio
```

## Usage After Implementation

```python
# Ollama - boolean
client.chat.completions.create(
    model="deepseek-r1",
    messages=[...],
    think=True,  # Enable thinking
)

# LM Studio with gpt-oss - intensity levels
client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[...],
    think="high",  # 'low', 'medium', or 'high'
)
```

## Testing Considerations

- Test with Ollama + DeepSeek R1 (boolean `think`)
- Test with LM Studio + gpt-oss (string intensity levels)
- Verify backward compatibility when `think` is not specified
- Test streaming mode with thinking enabled
- Verify thinking content is properly separated in response

## References

- [Ollama Thinking Blog Post](https://ollama.com/blog/thinking)
- [LangChain Ollama ChatModel Reference](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html)
- [LM Studio GPT-OSS Thinking Issue #851](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/851)
