"""Contextual Retrieval - LLM-generated context for chunks.

Implements Anthropic's Contextual Retrieval approach:
https://www.anthropic.com/news/contextual-retrieval

Key idea: Prepend LLM-generated context to each chunk before embedding,
reducing retrieval failures by ~40-50% when combined with hybrid search + reranking.

The context situates the chunk within the overall document, e.g.:
"This chunk is from the Configuration section of the Vault PKI documentation.
It describes certificate authority setup options."
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from tqdm import tqdm

from .config import RAGConfig

if TYPE_CHECKING:
    from llm_tools_server.config import ServerConfig

logger = logging.getLogger(__name__)


class ChunkContextualizer:
    """Generate and cache contextual prefixes for document chunks using local LLM.

    Uses the same backend (LM Studio or Ollama) configured for the main server,
    or can be overridden via RAGConfig contextual_backend_* settings.
    """

    def __init__(self, config: RAGConfig, cache_dir: Path, server_config: ServerConfig | None = None):
        """Initialize the contextualizer.

        Args:
            config: RAG configuration with contextual retrieval settings
            cache_dir: Directory to cache contextualized chunks
            server_config: Optional ServerConfig to use for backend settings.
                          If not provided, RAGConfig contextual_* settings must be set.
        """
        self.config = config
        self.server_config = server_config
        self.cache_dir = cache_dir
        self.context_cache_file = cache_dir / "chunk_contexts.json"

        # Resolve backend settings (RAGConfig overrides take precedence, then ServerConfig)
        self._resolve_backend_settings()

        # Load existing context cache
        self.context_cache: dict[str, str] = self._load_context_cache()

        # Pause control for background processing - set when user request starts, cleared when done
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused (set = running, clear = paused)

        # Stop control for graceful shutdown
        self._stop_event = threading.Event()

    def _resolve_backend_settings(self):
        """Resolve backend type, endpoint, and model from config hierarchy."""
        # Warn if contextual retrieval is enabled but no server_config provided
        missing_backend_config = not (self.config.contextual_backend_type and self.config.contextual_backend_endpoint)
        if self.config.contextual_retrieval_enabled and not self.server_config and missing_backend_config:
            logger.warning(
                "[RAG] Contextual retrieval enabled but no server_config provided. "
                "Pass server_config to DocSearchIndex() or set contextual_backend_type/endpoint in RAGConfig. "
                "Falling back to Ollama at localhost:11434."
            )
            print(
                "[RAG] WARNING: Contextual retrieval enabled but no server_config provided. "
                "Falling back to Ollama at localhost:11434.",
                file=sys.stderr,
            )

        # Backend type: RAGConfig override > ServerConfig > fallback
        if self.config.contextual_backend_type:
            self.backend_type = self.config.contextual_backend_type
        elif self.server_config:
            self.backend_type = self.server_config.BACKEND_TYPE
        else:
            self.backend_type = "ollama"  # Fallback default

        # Backend endpoint: RAGConfig override > ServerConfig > defaults
        if self.config.contextual_backend_endpoint:
            self.backend_endpoint = self.config.contextual_backend_endpoint
        elif self.server_config:
            if self.backend_type == "lmstudio":
                self.backend_endpoint = self.server_config.LMSTUDIO_ENDPOINT
            else:
                self.backend_endpoint = self.server_config.OLLAMA_ENDPOINT
        else:
            # Defaults if no server config
            if self.backend_type == "lmstudio":
                self.backend_endpoint = "http://localhost:1234/v1"
            else:
                self.backend_endpoint = "http://localhost:11434"

        # Model: RAGConfig override > ServerConfig > error if enabled
        if self.config.contextual_model:
            self.model = self.config.contextual_model
        elif self.server_config:
            self.model = self.server_config.BACKEND_MODEL
        else:
            self.model = None  # Will error if contextual retrieval is enabled

        if self.config.contextual_retrieval_enabled:
            logger.info(
                f"[RAG] Contextual retrieval using {self.backend_type} backend "
                f"at {self.backend_endpoint} with model {self.model}"
            )
            print(
                f"[RAG] Contextual retrieval using {self.backend_type} backend "
                f"at {self.backend_endpoint} with model {self.model}",
                file=sys.stderr,
            )

    def contextualize_chunks(
        self,
        chunks: list[dict[str, Any]],
        page_contents: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Add contextual prefixes to chunks using local LLM.

        Args:
            chunks: List of chunk dicts with 'content', 'chunk_id', and 'url' or parent with 'url'
            page_contents: Dict mapping URL -> full page text content

        Returns:
            Chunks with contextualized content (context prepended)
        """
        if not self.config.contextual_retrieval_enabled:
            return chunks

        logger.info(f"[RAG] Contextualizing {len(chunks)} chunks using {self.config.contextual_model}...")
        start_time = time.time()

        # Identify chunks that need context generation (not in cache)
        chunks_to_process = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            content_hash = self._hash_content(chunk.get("content", ""))

            # Check cache by chunk_id + content hash
            cache_key = f"{chunk_id}:{content_hash}"
            if cache_key not in self.context_cache:
                chunks_to_process.append((chunk, cache_key))

        cached_count = len(chunks) - len(chunks_to_process)
        if cached_count > 0:
            logger.info(
                f"[RAG] Using cached context for {cached_count} chunks, generating for {len(chunks_to_process)}"
            )

        # Generate context for uncached chunks in parallel
        if chunks_to_process:
            self._generate_contexts_parallel(chunks_to_process, page_contents)
            # Save updated cache
            self._save_context_cache()

        # Apply context to all chunks
        contextualized_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            content_hash = self._hash_content(chunk.get("content", ""))
            cache_key = f"{chunk_id}:{content_hash}"

            context = self.context_cache.get(cache_key, "")
            if context:
                # Prepend context to chunk content
                contextualized_content = f"<context>\n{context}\n</context>\n\n{chunk['content']}"
                contextualized_chunk = {
                    **chunk,
                    "content": contextualized_content,
                    "original_content": chunk["content"],
                }
            else:
                contextualized_chunk = chunk

            contextualized_chunks.append(contextualized_chunk)

        elapsed = time.time() - start_time
        logger.info(f"[RAG] Contextualization complete in {elapsed:.1f}s")

        return contextualized_chunks

    def _generate_contexts_parallel(
        self,
        chunks_to_process: list[tuple[dict[str, Any], str]],
        page_contents: dict[str, str],
        save_every: int = 50,
    ):
        """Generate contexts for chunks in parallel using ThreadPoolExecutor.

        Progress is saved incrementally to allow resumption if interrupted.

        Args:
            chunks_to_process: List of (chunk, cache_key) tuples
            page_contents: Dict mapping URL -> full page text content
            save_every: Save cache every N completed chunks (default: 50)
        """
        failed = 0
        last_save = 0

        with ThreadPoolExecutor(max_workers=self.config.contextual_max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {}
            for chunk, cache_key in chunks_to_process:
                url = chunk.get("url") or chunk.get("metadata", {}).get("url", "")
                document_content = page_contents.get(url, "")

                if not document_content:
                    logger.warning(f"[RAG] No document content found for URL: {url}")
                    continue

                future = executor.submit(
                    self._generate_single_context,
                    chunk["content"],
                    document_content,
                )
                future_to_chunk[future] = (chunk, cache_key)

            # Process completed tasks with progress bar
            with tqdm(
                total=len(future_to_chunk),
                desc="Generating contexts",
                unit="chunks",
                file=sys.stderr,
            ) as pbar:
                for future in as_completed(future_to_chunk):
                    chunk, cache_key = future_to_chunk[future]

                    try:
                        context = future.result()
                        if context:
                            self.context_cache[cache_key] = context
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"[RAG] Context generation failed for chunk {chunk.get('chunk_id')}: {e}")
                        failed += 1

                    pbar.update(1)
                    if failed > 0:
                        pbar.set_postfix(failed=failed)

                    # Save cache periodically to allow resumption
                    if pbar.n - last_save >= save_every:
                        self._save_context_cache()
                        last_save = pbar.n
                        logger.debug(f"[RAG] Saved context cache checkpoint ({pbar.n} chunks)")

        # Final save
        if len(future_to_chunk) > last_save:
            self._save_context_cache()

    def _generate_single_context(self, chunk_content: str, document_content: str) -> str | None:
        """Generate context for a single chunk using the configured backend.

        Supports both LM Studio (OpenAI-compatible) and Ollama backends.
        Respects pause/resume for yielding to user requests.

        Args:
            chunk_content: The chunk text to contextualize
            document_content: The full document text

        Returns:
            Generated context string, or None if failed
        """
        # Check if we should stop
        if self._stop_event.is_set():
            return None

        # Wait if paused (yields to user requests)
        # Uses timeout to periodically check - blocks until resumed or stopped
        while not self._pause_event.wait(timeout=0.5):
            if self._stop_event.is_set():
                return None

        # Check again after waking up
        if self._stop_event.is_set():
            return None

        # Build prompt from template
        prompt = self.config.contextual_prompt.format(
            document=document_content,
            chunk=chunk_content,
        )

        try:
            if self.backend_type == "lmstudio":
                # LM Studio uses OpenAI-compatible chat completions API
                response = requests.post(
                    f"{self.backend_endpoint}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "stream": False,
                    },
                    timeout=self.config.contextual_timeout,
                )
                response.raise_for_status()
                result = response.json()
                context = result["choices"][0]["message"]["content"].strip()
            else:
                # Ollama uses its native /api/generate endpoint
                response = requests.post(
                    f"{self.backend_endpoint}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=self.config.contextual_timeout,
                )
                response.raise_for_status()
                result = response.json()
                context = result.get("response", "").strip()

            # Basic validation - context should be reasonable length
            if len(context) < 10:
                logger.warning(f"[RAG] Generated context too short: {context[:50]}")
                return None

            if len(context) > 1000:
                # Truncate overly long contexts
                context = context[:1000] + "..."
                logger.debug("[RAG] Truncated long context to 1000 chars")

            return context

        except requests.exceptions.Timeout:
            logger.warning(f"[RAG] Context generation timed out after {self.config.contextual_timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[RAG] Backend request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[RAG] Context generation error: {e}")
            return None

    def _hash_content(self, content: str) -> str:
        """Generate hash of content for cache key."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_context_cache(self) -> dict[str, str]:
        """Load context cache from disk."""
        if not self.context_cache_file.exists():
            return {}

        try:
            cache = json.loads(self.context_cache_file.read_text())
            logger.info(f"[RAG] Loaded {len(cache)} cached chunk contexts")
            return cache
        except Exception as e:
            logger.warning(f"[RAG] Failed to load context cache: {e}")
            return {}

    def _save_context_cache(self):
        """Save context cache to disk."""
        try:
            self.context_cache_file.write_text(json.dumps(self.context_cache, indent=2))
            logger.debug(f"[RAG] Saved {len(self.context_cache)} chunk contexts to cache")
        except Exception as e:
            logger.error(f"[RAG] Failed to save context cache: {e}")

    def clear_cache(self):
        """Clear the context cache (forces regeneration on next run)."""
        self.context_cache = {}
        if self.context_cache_file.exists():
            self.context_cache_file.unlink()
        logger.info("[RAG] Context cache cleared")

    def pause(self):
        """Pause background context generation (call when user request starts)."""
        self._pause_event.clear()
        logger.debug("[RAG] Contextual retrieval paused for user request")

    def resume(self):
        """Resume background context generation (call when user request completes)."""
        self._pause_event.set()
        logger.debug("[RAG] Contextual retrieval resumed")

    def is_paused(self) -> bool:
        """Check if context generation is currently paused."""
        return not self._pause_event.is_set()

    def stop(self):
        """Stop background context generation for graceful shutdown.

        Sets the stop flag and wakes up any paused workers so they can exit.
        """
        self._stop_event.set()
        self._pause_event.set()  # Wake up any paused workers
        logger.info("[RAG] Contextual retrieval stop requested")

    def is_stopped(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()
