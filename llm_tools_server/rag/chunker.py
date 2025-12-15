"""Parent-Child Semantic HTML Chunking.

Token-aware HTML chunking that respects document structure (headings) and creates
hierarchical parent-child relationships for precise retrieval + rich context.

Key Features:
- Token-aware limits using tiktoken (not character counts)
- Level stack DOM hierarchy tracking
- Atomic handling of code blocks and tables
- Boilerplate removal (nav, breadcrumbs, buttons)
- Metadata extraction from URL and content structure
"""

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any

import tiktoken
from bs4 import BeautifulSoup, NavigableString

# Initialize tokenizer (cl100k_base is used by GPT-4, GPT-3.5-turbo)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Boilerplate CSS selectors to skip
BOILERPLATE_SELECTORS = [
    "nav",
    "aside",
    "footer",
    "header",
    '[role="navigation"]',
    '[role="banner"]',
    '[role="contentinfo"]',
    '[aria-label*="breadcrumb"]',
    '[aria-label*="navigation"]',
    ".edit-on-github",
    ".copy-button",
    ".toc",
    ".table-of-contents",
    ".nav",
    ".navbar",
    ".sidebar",
    ".footer",
    ".header",
    'button[class*="copy"]',
    'a[class*="edit"]',
]

# Minimum content length in characters to create a chunk.
# Content blocks shorter than this are skipped to avoid tiny, low-quality fragments
# like navigation text, button labels, or isolated punctuation. The threshold of 20
# characters was chosen empirically to filter noise while preserving meaningful
# short content like code snippets, definitions, or brief paragraphs.
MIN_CONTENT_LENGTH = 20

# Ratio of child_min_tokens used as threshold for merged small content blocks.
# When accumulating small fragments that individually are too small to be chunks,
# we use this lower threshold (child_min_tokens * MERGED_CONTENT_THRESHOLD_RATIO)
# to decide if the merged result is worth keeping. Set to 0.5 (50%) to allow
# smaller merged chunks than normal minimum, while still filtering very tiny ones.
MERGED_CONTENT_THRESHOLD_RATIO = 0.5


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""

    heading_path: list[str]
    heading_path_joined: str
    section_id: str
    url: str
    doc_type: str
    version: str | None
    code_identifiers: str
    is_parent: bool
    parent_id: str | None = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    if not text:
        return 0
    return len(tokenizer.encode(text, disallowed_special=()))


def semantic_chunk_html(
    html: str,
    url: str,
    child_min_tokens: int = 150,
    child_max_tokens: int = 350,
    parent_min_tokens: int = 300,
    parent_max_tokens: int = 900,
    absolute_max_tokens: int = 1200,
) -> dict[str, list[dict[str, Any]]]:
    """Chunk HTML into parent-child structure using heading hierarchy.

    Args:
        html: HTML content to chunk
        url: Source URL
        child_min_tokens: Minimum tokens for child chunk
        child_max_tokens: Maximum tokens for child chunk
        parent_min_tokens: Minimum tokens for parent chunk
        parent_max_tokens: Maximum tokens for parent chunk
        absolute_max_tokens: Absolute maximum tokens (split if exceeded)

    Returns:
        Dictionary with 'parents' and 'children' lists
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove boilerplate elements that may remain after readability extraction
    # (especially when fallback to <main>/<article> tags or original HTML is used)
    for selector in BOILERPLATE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    # Use the body or the full soup as the content container
    content_container = soup.body or soup

    if not content_container:
        return {"parents": [], "children": []}

    parents = []
    children = []

    # Track heading hierarchy with level stack
    heading_stack = []

    # Initialize with a root section for content before the first heading
    # This ensures intro/overview content is captured
    current_section = {
        "heading_stack": [],  # Empty heading stack for root section
        "content_blocks": [],
        "code_blocks": [],
        "tables": [],
    }

    # Walk through all descendants to get content
    processed_elements = set()

    for elem in content_container.descendants:
        # Skip NavigableString that are just whitespace
        if isinstance(elem, NavigableString):
            if not elem.strip():
                continue
            # If we have a current section and this is meaningful text, add it
            if current_section and elem.strip():
                current_section["content_blocks"].append(elem.strip())
            continue

        # Skip if already processed
        if id(elem) in processed_elements:
            continue
        processed_elements.add(id(elem))

        if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Finalize previous section if exists
            if current_section:
                _finalize_section(
                    current_section,
                    parents,
                    children,
                    url,
                    heading_stack,
                    child_min_tokens,
                    child_max_tokens,
                    parent_min_tokens,
                    parent_max_tokens,
                    absolute_max_tokens,
                )

            # Update heading stack (pop levels >= current level)
            level = int(elem.name[1])
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()

            # Add current heading to stack
            heading_text = elem.get_text().strip()
            heading_id = elem.get("id", _slugify(heading_text))

            heading_stack.append({"level": level, "text": heading_text, "id": heading_id})

            # Start new section
            current_section = {
                "heading_stack": list(heading_stack),  # Copy current stack
                "content_blocks": [],
                "code_blocks": [],
                "tables": [],
            }

        elif elem.name == "pre":
            # Code block (often contains <code> inside)
            if current_section:
                code = elem.find("code")
                if code:
                    lang = code.get("class", [""])[0].replace("language-", "") if code.get("class") else ""
                    current_section["code_blocks"].append({"lang": lang, "code": code.get_text()})
                else:
                    current_section["code_blocks"].append({"lang": "", "code": elem.get_text()})

        elif elem.name == "table":
            # Table - keep atomic
            if current_section:
                table_text = elem.get_text(separator=" ", strip=True)
                current_section["tables"].append(table_text)

        elif elem.name in ["p", "div", "ul", "ol", "dl", "blockquote"]:
            # Content blocks
            if current_section:
                text = elem.get_text(separator=" ", strip=True)
                if text:  # Only add non-empty content
                    current_section["content_blocks"].append(text)

    # Finalize last section
    if current_section:
        _finalize_section(
            current_section,
            parents,
            children,
            url,
            heading_stack,
            child_min_tokens,
            child_max_tokens,
            parent_min_tokens,
            parent_max_tokens,
            absolute_max_tokens,
        )

    return {"parents": parents, "children": children}


def _finalize_section(
    section: dict[str, Any],
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    url: str,
    heading_stack: list[dict[str, str]],
    child_min_tokens: int,
    child_max_tokens: int,
    parent_min_tokens: int,
    parent_max_tokens: int,
    absolute_max_tokens: int,
):
    """Create parent chunk and child chunks from a section."""
    # Build heading path
    heading_path = [h["text"] for h in section["heading_stack"]]
    section_id = section["heading_stack"][-1]["id"] if section["heading_stack"] else "root"

    # Combine all content
    all_text_parts = []

    # Add content blocks (paragraphs, lists, etc.)
    for content in section.get("content_blocks", []):
        if content and len(content.strip()) > MIN_CONTENT_LENGTH:
            all_text_parts.append(content.strip())

    # Add code blocks with markdown formatting
    for code_data in section.get("code_blocks", []):
        lang = code_data.get("lang", "text")
        code = code_data.get("code", "").strip()
        if code:
            all_text_parts.append(f"```{lang}\n{code}\n```")

    # Add tables
    for table_text in section.get("tables", []):
        if table_text and len(table_text.strip()) > 20:
            all_text_parts.append(table_text.strip())

    if not all_text_parts:
        return  # Empty section, skip

    all_text = "\n\n".join(all_text_parts)
    tokens = count_tokens(all_text)

    canonical_url = _canonicalize_url(url)

    # Extract metadata
    metadata = _build_metadata(
        heading_path=heading_path,
        section_id=section_id,
        url=canonical_url,
        content=all_text,
        code_blocks=section.get("code_blocks", []),
    )

    # Create parent chunk(s)
    if tokens <= parent_max_tokens:
        # Section fits in one parent
        parent_id = _generate_chunk_id(canonical_url, heading_path, 0)

        parent_chunk = {
            "chunk_id": parent_id,
            "content": all_text,
            "tokens": tokens,
            "metadata": replace(metadata, is_parent=True),
        }
        parents.append(parent_chunk)

        # Create child chunks from this parent
        _create_children_from_section(section, parent_id, children, metadata, child_min_tokens, child_max_tokens)

    else:
        # Split large section on paragraph/block boundaries
        _split_large_section(
            all_text_parts,
            canonical_url,
            heading_path,
            metadata,
            parents,
            children,
            child_min_tokens,
            child_max_tokens,
            parent_max_tokens,
            absolute_max_tokens,
        )


def _split_large_section(
    text_parts: list[str],
    canonical_url: str,
    heading_path: list[str],
    metadata: ChunkMetadata,
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    child_min_tokens: int,
    child_max_tokens: int,
    parent_max_tokens: int,
    absolute_max_tokens: int,
):
    """Split a large section into multiple parent chunks on paragraph boundaries."""
    current_parts = []
    current_tokens = 0
    sub_idx = 0

    for part in text_parts:
        part_tokens = count_tokens(part)

        # If single part exceeds absolute max, we need to split it further
        if part_tokens > absolute_max_tokens:
            # Flush current if exists
            if current_parts:
                _flush_parent_chunk(
                    current_parts,
                    canonical_url,
                    heading_path,
                    sub_idx,
                    metadata,
                    parents,
                    children,
                    child_min_tokens,
                    child_max_tokens,
                )
                current_parts = []
                current_tokens = 0
                sub_idx += 1

            # Split the oversized part on sentence boundaries
            _split_oversized_part(
                part,
                canonical_url,
                heading_path,
                sub_idx,
                metadata,
                parents,
                children,
                child_min_tokens,
                child_max_tokens,
                parent_max_tokens,
            )
            sub_idx += 1
            continue

        # Check if adding this part would exceed limit
        if current_tokens + part_tokens > parent_max_tokens and current_parts:
            # Flush current parent
            _flush_parent_chunk(
                current_parts,
                canonical_url,
                heading_path,
                sub_idx,
                metadata,
                parents,
                children,
                child_min_tokens,
                child_max_tokens,
            )
            current_parts = [part]
            current_tokens = part_tokens
            sub_idx += 1
        else:
            current_parts.append(part)
            current_tokens += part_tokens

    # Flush final parent
    if current_parts:
        _flush_parent_chunk(
            current_parts,
            canonical_url,
            heading_path,
            sub_idx,
            metadata,
            parents,
            children,
            child_min_tokens,
            child_max_tokens,
        )


def _flush_parent_chunk(
    text_parts: list[str],
    canonical_url: str,
    heading_path: list[str],
    sub_idx: int,
    metadata: ChunkMetadata,
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    child_min_tokens: int,
    child_max_tokens: int,
):
    """Create a parent chunk and its children from text parts.

    Handles content of any size - small parts are merged, large parts are split.
    """
    content = "\n\n".join(text_parts)
    tokens = count_tokens(content)
    parent_id = _generate_chunk_id(canonical_url, heading_path, sub_idx)

    parent_chunk = {
        "chunk_id": parent_id,
        "content": content,
        "tokens": tokens,
        "metadata": replace(metadata, is_parent=True),
        "sub_chunk": sub_idx,
    }
    parents.append(parent_chunk)

    # Create children from text parts with proper handling of all sizes
    child_idx = 0
    pending_content = []
    pending_tokens = 0

    def add_child(text: str, tok: int):
        nonlocal child_idx
        child_chunk = {
            "chunk_id": _generate_chunk_id(canonical_url, heading_path, f"{sub_idx}_child_{child_idx}"),
            "parent_id": parent_id,
            "content": text,
            "tokens": tok,
            "metadata": replace(metadata, is_parent=False, parent_id=parent_id),
        }
        children.append(child_chunk)
        child_idx += 1

    def flush_pending():
        nonlocal pending_content, pending_tokens
        if pending_content and pending_tokens >= child_min_tokens // 2:
            merged = "\n\n".join(pending_content)
            add_child(merged, pending_tokens)
        pending_content = []
        pending_tokens = 0

    for part in text_parts:
        part_tokens = count_tokens(part)

        if part_tokens > child_max_tokens:
            # Split large parts at sentence boundaries
            flush_pending()
            sentences = re.split(r"(?<=[.!?])\s+", part)
            current_text = ""
            current_tokens = 0
            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if current_tokens + sent_tokens > child_max_tokens and current_text:
                    add_child(current_text.strip(), current_tokens)
                    current_text = sentence
                    current_tokens = sent_tokens
                else:
                    current_text += " " + sentence if current_text else sentence
                    current_tokens += sent_tokens
            if current_text.strip():
                if current_tokens >= child_min_tokens:
                    add_child(current_text.strip(), current_tokens)
                else:
                    pending_content.append(current_text.strip())
                    pending_tokens += current_tokens
        elif part_tokens >= child_min_tokens:
            flush_pending()
            add_child(part, part_tokens)
        else:
            # Accumulate small parts
            pending_content.append(part)
            pending_tokens += part_tokens
            if pending_tokens >= child_min_tokens:
                flush_pending()

    flush_pending()


def _split_oversized_part(
    text: str,
    canonical_url: str,
    heading_path: list[str],
    sub_idx: int,
    metadata: ChunkMetadata,
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    child_min_tokens: int,
    child_max_tokens: int,
    parent_max_tokens: int,
):
    """Split an oversized text part on sentence boundaries.

    Creates parent chunks that fit within parent_max_tokens, and child chunks
    for all content (splitting large chunks, merging small ones).
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    current_text = ""
    current_tokens = 0
    chunk_idx = 0

    def create_parent_and_children(content: str, tokens: int, idx: int):
        """Create a parent chunk and appropriate child chunks."""
        parent_id = _generate_chunk_id(canonical_url, heading_path, f"{sub_idx}_{idx}")
        parent_chunk = {
            "chunk_id": parent_id,
            "content": content,
            "tokens": tokens,
            "metadata": replace(metadata, is_parent=True),
            "sub_chunk": f"{sub_idx}_{idx}",
        }
        parents.append(parent_chunk)

        # Create child chunk(s) - split if too large, always create if meaningful
        if tokens > child_max_tokens:
            # Split into multiple children at sentence boundaries
            child_sentences = re.split(r"(?<=[.!?])\s+", content)
            child_text = ""
            child_tokens = 0
            child_idx = 0
            for sent in child_sentences:
                sent_tok = count_tokens(sent)
                if child_tokens + sent_tok > child_max_tokens and child_text:
                    child_chunk = {
                        "chunk_id": _generate_chunk_id(
                            canonical_url, heading_path, f"{sub_idx}_{idx}_child_{child_idx}"
                        ),
                        "parent_id": parent_id,
                        "content": child_text.strip(),
                        "tokens": child_tokens,
                        "metadata": replace(metadata, is_parent=False, parent_id=parent_id),
                    }
                    children.append(child_chunk)
                    child_idx += 1
                    child_text = sent
                    child_tokens = sent_tok
                else:
                    child_text += " " + sent if child_text else sent
                    child_tokens += sent_tok
            if child_text.strip():
                child_chunk = {
                    "chunk_id": _generate_chunk_id(canonical_url, heading_path, f"{sub_idx}_{idx}_child_{child_idx}"),
                    "parent_id": parent_id,
                    "content": child_text.strip(),
                    "tokens": child_tokens,
                    "metadata": replace(metadata, is_parent=False, parent_id=parent_id),
                }
                children.append(child_chunk)
        else:
            # Content fits in one child - create it regardless of min threshold
            # (parent already exists, child provides focused search target)
            child_chunk = {
                "chunk_id": _generate_chunk_id(canonical_url, heading_path, f"{sub_idx}_{idx}_child"),
                "parent_id": parent_id,
                "content": content,
                "tokens": tokens,
                "metadata": replace(metadata, is_parent=False, parent_id=parent_id),
            }
            children.append(child_chunk)

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        if current_tokens + sent_tokens > parent_max_tokens and current_text:
            # Flush current chunk
            create_parent_and_children(current_text, current_tokens, chunk_idx)
            current_text = sentence
            current_tokens = sent_tokens
            chunk_idx += 1
        else:
            current_text += " " + sentence if current_text else sentence
            current_tokens += sent_tokens

    # Flush final chunk
    if current_text:
        create_parent_and_children(current_text, current_tokens, chunk_idx)


def _create_children_from_section(
    section: dict[str, Any],
    parent_id: str,
    children: list[dict[str, Any]],
    metadata: ChunkMetadata,
    child_min_tokens: int,
    child_max_tokens: int,
):
    """Create child chunks from section content blocks.

    Handles content of any size:
    - Small content (<min_tokens): Accumulate and merge with adjacent content
    - Medium content (min-max tokens): Create child chunk directly
    - Large content (>max_tokens): Split at sentence boundaries
    """
    child_idx = 0

    # Accumulator for small content blocks that need merging
    pending_content = []
    pending_tokens = 0

    def flush_pending():
        """Flush accumulated small content as a child chunk."""
        nonlocal child_idx, pending_content, pending_tokens
        if not pending_content:
            return
        merged = "\n\n".join(pending_content)
        # Only create child if we have meaningful content
        if pending_tokens >= int(child_min_tokens * MERGED_CONTENT_THRESHOLD_RATIO):
            child_chunk = {
                "chunk_id": f"{parent_id}_child_{child_idx}",
                "parent_id": parent_id,
                "content": merged,
                "tokens": pending_tokens,
                "metadata": replace(metadata, is_parent=False, parent_id=parent_id),
            }
            children.append(child_chunk)
            child_idx += 1
        pending_content = []
        pending_tokens = 0

    def add_child(content: str, tokens: int):
        """Add a single child chunk."""
        nonlocal child_idx
        child_chunk = {
            "chunk_id": f"{parent_id}_child_{child_idx}",
            "parent_id": parent_id,
            "content": content,
            "tokens": tokens,
            "metadata": replace(metadata, is_parent=False, parent_id=parent_id),
        }
        children.append(child_chunk)
        child_idx += 1

    def split_large_content(content: str, tokens: int):
        """Split oversized content at sentence boundaries."""
        nonlocal child_idx
        sentences = re.split(r"(?<=[.!?])\s+", content)
        current_text = ""
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = count_tokens(sentence)

            if current_tokens + sent_tokens > child_max_tokens and current_text:
                # Flush current chunk
                add_child(current_text.strip(), current_tokens)
                current_text = sentence
                current_tokens = sent_tokens
            else:
                current_text += " " + sentence if current_text else sentence
                current_tokens += sent_tokens

        # Flush remaining
        if current_text.strip():
            if current_tokens >= child_min_tokens:
                add_child(current_text.strip(), current_tokens)
            else:
                # Add to pending for merging
                pending_content.append(current_text.strip())
                nonlocal pending_tokens
                pending_tokens += current_tokens

    # Process content blocks (paragraphs, lists)
    for content in section.get("content_blocks", []):
        if not content or len(content.strip()) < 20:
            continue

        tokens = count_tokens(content)

        if tokens > child_max_tokens:
            # Large content: flush pending first, then split
            flush_pending()
            split_large_content(content, tokens)
        elif tokens >= child_min_tokens:
            # Medium content: flush pending, then add directly
            flush_pending()
            add_child(content, tokens)
        else:
            # Small content: accumulate for merging
            pending_content.append(content)
            pending_tokens += tokens
            # If accumulated enough, flush
            if pending_tokens >= child_min_tokens:
                flush_pending()

    # Process code blocks (keep atomic, don't split)
    for code_data in section.get("code_blocks", []):
        lang = code_data.get("lang", "text")
        code = code_data.get("code", "").strip()
        if not code:
            continue

        code_text = f"```{lang}\n{code}\n```"
        tokens = count_tokens(code_text)

        # Flush any pending content before code
        flush_pending()

        # Code blocks are kept atomic - add regardless of size
        # (splitting code would break it)
        add_child(code_text, tokens)

    # Process tables (keep atomic, don't split)
    for table_text in section.get("tables", []):
        if not table_text or len(table_text.strip()) < 20:
            continue

        tokens = count_tokens(table_text)

        # Flush any pending content before table
        flush_pending()

        # Tables are kept atomic - add regardless of size
        add_child(table_text, tokens)

    # Flush any remaining pending content
    flush_pending()


def _generate_chunk_id(canonical_url: str, heading_path: list[str], sub_idx: Any) -> str:
    """Generate stable chunk ID from URL, heading path, and sub-index."""
    path_str = " > ".join(heading_path) if heading_path else "root"
    combined = f"{canonical_url}||{path_str}::{sub_idx}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _canonicalize_url(url: str) -> str:
    """Normalize URL to canonical form."""
    # Remove query params, trailing slash, anchors
    url = url.split("?")[0].split("#")[0].rstrip("/")
    return url


def _slugify(text: str) -> str:
    """Convert heading text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text


def _extract_version(heading_path: list[str], content: str) -> str | None:
    """Extract version number from heading or content."""
    # Look for patterns like "1.20.0", "v1.20", "1.20.x"
    version_pattern = r"v?(\d+\.\d+(?:\.\d+)?(?:\.x)?)"

    # Check headings first
    for heading in heading_path:
        match = re.search(version_pattern, heading)
        if match:
            return match.group(1)

    # Check first 200 chars of content
    match = re.search(version_pattern, content[:200])
    if match:
        return match.group(1)

    return None


def _detect_doc_type(url: str) -> str:
    """Detect document type from URL patterns."""
    url_lower = url.lower()

    if "/release-notes" in url_lower or "/updates" in url_lower or "/releases" in url_lower:
        return "release-notes"
    elif "/api" in url_lower or "/api-docs" in url_lower:
        return "api"
    elif "/tutorials" in url_lower or "/tutorial" in url_lower:
        return "tutorial"
    elif "/commands/" in url_lower or "/cli" in url_lower:
        return "cli"
    elif "/docs/" in url_lower or "/documentation" in url_lower:
        return "docs"
    elif "/guide" in url_lower or "/guides" in url_lower:
        return "guide"
    elif "/reference" in url_lower:
        return "reference"
    else:
        return "docs"


def _extract_code_identifiers(code_blocks: list[dict[str, str]]) -> str:
    """Extract function names, variables, flags from code blocks."""
    identifiers = set()

    for code_data in code_blocks:
        code = code_data.get("code", "")

        # Extract function names (e.g., function_name(...))
        funcs = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", code, re.IGNORECASE)
        identifiers.update(funcs)

        # Extract CLI flags (e.g., --flag-name)
        flags = re.findall(r"--([a-z][a-z0-9-]+)", code)
        identifiers.update(flags)

        # Extract env vars (e.g., VAULT_ADDR)
        env_vars = re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", code)
        identifiers.update(env_vars)

    # Return top 20 most common identifiers
    return " ".join(sorted(identifiers)[:20])


def _build_metadata(
    heading_path: list[str], section_id: str, url: str, content: str, code_blocks: list[dict[str, str]]
) -> ChunkMetadata:
    """Build metadata for a chunk."""
    return ChunkMetadata(
        heading_path=heading_path,
        heading_path_joined=" > ".join(heading_path) if heading_path else "",
        section_id=section_id,
        url=url,
        doc_type=_detect_doc_type(url),
        version=_extract_version(heading_path, content),
        code_identifiers=_extract_code_identifiers(code_blocks),
        is_parent=True,  # Will be overridden for children
        parent_id=None,
    )
