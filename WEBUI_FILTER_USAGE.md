# Open Web UI Query Filter - Usage Guide

## Problem

Your tool logs show that RAG retrieval is happening for **both**:
1. Real user queries: `"do you get anything at checkin for being Marriott titanium"`
2. System tasks: `"### Task:\nSuggest 3-5 relevant follow-up questions..."`

System tasks don't need RAG context, so you're wasting:
- API calls
- Latency (500ms+ per RAG retrieval)
- Costs
- Log clutter

## Solution

Use the new `is_webui_system_task()` filter to skip RAG for system tasks.

## Implementation (milesoss)

### 1. Find your RAG tool

Look for the tool that does RAG retrieval in your milesoss project. Based on your logs, it appears to be logging `"rag_augmentation"` events.

### 2. Add the filter

```python
from llm_api_server import is_webui_system_task

def your_rag_tool(query: str) -> str:
    """Your RAG retrieval tool."""

    # NEW: Skip RAG for Open Web UI system tasks
    if is_webui_system_task(query):
        self.logger.debug({
            "event": "rag_skipped",
            "reason": "webui_system_task",
            "query": query[:100],  # Truncated for logging
        })
        return ""  # No context needed

    # EXISTING: Do RAG retrieval for real queries
    self.logger.debug({
        "event": "rag_augmentation",
        "user_query": query,
        # ... rest of your logging
    })

    results = self.index.search(query, top_k=5)
    context = format_context(results)

    return context
```

### 3. Test it

After making the change, your logs should show:

**Before (wasteful):**
```
2025-11-25 20:16:23 - miles.tools - DEBUG - {
  "event": "rag_augmentation",
  "user_query": "### Task:\nSuggest 3-5 relevant follow-up questions",
  "rag_context": "Here is some relevant context...",  # ← WASTED
  "context_length": 1876
}
```

**After (efficient):**
```
2025-11-25 20:16:23 - miles.tools - DEBUG - {
  "event": "rag_skipped",
  "reason": "webui_system_task",
  "query": "### Task:\nSuggest 3-5 relevant follow-up questions"
}
```

### 4. Expected Impact

Based on your logs, approximately **50% of queries** are system tasks:
- 3 real queries: Marriott titanium, room availability, Hyatt tiers
- 6 system tasks: 2× follow-ups, 2× titles, 2× tags

**Savings:**
- 50% fewer RAG API calls
- 50% reduction in RAG latency
- Cleaner logs (system tasks clearly identified)

## Advanced: Custom Patterns

If you want to filter additional patterns:

```python
from llm_api_server.webui_utils import WebUIQueryFilter

# Create custom filter
filter = WebUIQueryFilter(
    patterns=[r"^CUSTOM: Generate"],  # Add your patterns
    use_defaults=True,  # Keep default Open Web UI patterns
)

# Use it
if filter.is_system_task(query):
    return ""
```

## Default Patterns Detected

The filter automatically detects these Open Web UI system tasks:
- Follow-up generation: `### Task:\nSuggest 3-5 relevant follow-up questions`
- Title generation: `### Task:\nGenerate a concise, 3-5 word title`
- Tag generation: `### Task:\nGenerate 1-3 broad tags categorizing`

## Testing

Run the example to see it in action:
```bash
uv run python examples/webui_rag_filter_example.py
```

## Configuration Option (Optional)

You could also add a config flag to make it optional:

```python
class MilesConfig(ServerConfig):
    SKIP_RAG_FOR_SYSTEM_TASKS: bool = True  # Default: enabled

# In your tool
if config.SKIP_RAG_FOR_SYSTEM_TASKS and is_webui_system_task(query):
    return ""
```

## Questions?

- See `llm_api_server/webui_utils.py` for implementation
- See `tests/test_webui_utils.py` for test cases
- See `examples/webui_rag_filter_example.py` for full examples
