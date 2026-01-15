"""
Document ingestion pipeline with contextual chunking.

Collects documents, chunks them with context, generates embeddings via Modal,
and stores in pgvector.

Usage:
    python ingest.py ~/notes ~/projects/docs
    python ingest.py ~/notes --metadata '{"project": "personal"}'
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import yaml
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .config import config


def read_text_with_fallback(path: Path, encodings: tuple[str, ...] = ("utf-8", "windows-1252", "latin-1")) -> str:
    """Read text file trying multiple encodings in order."""
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Last resort: read with errors replaced
    return path.read_text(encoding="utf-8", errors="replace")


def matches_pattern(filename: str, patterns: list[str]) -> str | None:
    """Check if filename matches any pattern. Returns matched pattern or None."""
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(filename.lower(), pattern.lower()):
            return pattern
    return None


# Patterns for detecting secrets in content
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]* PRIVATE KEY-----"), "private key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "GitHub personal access token"),
    (re.compile(r"gho_[a-zA-Z0-9]{36}"), "GitHub OAuth token"),
    (re.compile(r"sk-[a-zA-Z0-9]{48}"), "OpenAI API key"),
    (re.compile(r"sk-ant-api[a-zA-Z0-9-]{80,}"), "Anthropic API key"),
]


def scan_content_for_secrets(content: str) -> str | None:
    """Scan content for potential secrets. Returns description if found, None otherwise."""
    # Only check first 10KB to avoid slow scans on large files
    sample = content[:10240]
    for pattern, description in SECRET_PATTERNS:
        if pattern.search(sample):
            return description
    return None


def is_minified(content: str, max_line_length: int = 1000) -> bool:
    """Detect if content appears to be minified JS/CSS."""
    lines = content.split("\n", 10)  # Only check first few lines
    if not lines:
        return False
    # Check if any of the first lines is extremely long
    for line in lines[:5]:
        if len(line) > max_line_length:
            # Also check it's not just a long string/comment - minified has lots of punctuation
            if line.count(";") > 20 or line.count(",") > 50 or line.count("{") > 20:
                return True
    return False


class FileSkipReason:
    """Result of file skip check."""
    def __init__(self, should_skip: bool, reason: str = "", is_security: bool = False):
        self.should_skip = should_skip
        self.reason = reason
        self.is_security = is_security  # True for blocked (security), False for skipped (low-value)


def check_file_skip(path: Path, content: str | None = None) -> FileSkipReason:
    """
    Check if a file should be skipped or blocked.

    Returns FileSkipReason with details.
    """
    filename = path.name

    # Check block patterns (security)
    if matched := matches_pattern(filename, config.block_patterns):
        return FileSkipReason(True, f"matches block pattern '{matched}'", is_security=True)

    # Check skip patterns (low-value)
    if matched := matches_pattern(filename, config.skip_patterns):
        return FileSkipReason(True, f"matches skip pattern '{matched}'", is_security=False)

    # Content-based checks (if content provided and scanning enabled)
    if content is not None and config.scan_content:
        # Check for secrets
        if secret_type := scan_content_for_secrets(content):
            return FileSkipReason(True, f"contains {secret_type}", is_security=True)

        # Check for minified JS/CSS
        if path.suffix in (".js", ".css") and is_minified(content, config.max_line_length_for_minified):
            return FileSkipReason(True, "appears to be minified", is_security=False)

    return FileSkipReason(False)


@dataclass
class DocumentMetadata:
    """Metadata extracted from document or provided externally."""

    tags: list[str] = field(default_factory=list)
    project: str | None = None
    category: str | None = None
    status: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_frontmatter(cls, frontmatter: dict) -> DocumentMetadata:
        """Create from YAML frontmatter."""
        extra = {
            k: v
            for k, v in frontmatter.items()
            if k not in {"tags", "project", "category", "status"}
        }
        if doc_date := extract_document_date(frontmatter):
            extra["document_date"] = doc_date
        return cls(
            tags=frontmatter.get("tags", []),
            project=frontmatter.get("project"),
            category=frontmatter.get("category"),
            status=frontmatter.get("status"),
            extra=extra,
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {}
        if self.tags:
            result["tags"] = self.tags
        if self.project:
            result["project"] = self.project
        if self.category:
            result["category"] = self.category
        if self.status:
            result["status"] = self.status
        if self.extra:
            result.update(self.extra)
        return result


@dataclass
class Document:
    """A document to be indexed."""

    source_path: str
    source_type: str
    title: str
    content: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    sections: list[tuple[str, str]] = field(default_factory=list)  # (header, content)


@dataclass
class Chunk:
    """A chunk ready for embedding."""

    content: str  # Original text (for display)
    embedding_text: str  # Contextualized text (for embedding)
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)


def content_hash(content: str) -> str:
    """Generate hash for deduplication/change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def extract_document_date(metadata: dict) -> str | None:
    """Extract document date from frontmatter/metadata, trying common field names."""
    date_fields = ["date", "created", "modified", "updated", "last_modified", "pubdate"]
    for field_name in date_fields:
        if value := metadata.get(field_name):
            if hasattr(value, "isoformat"):
                return value.isoformat()
            if isinstance(value, str):
                return value
    return None


def extract_frontmatter(content: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter from markdown content.

    Returns (frontmatter_dict, remaining_content).
    """
    if not content.startswith("---"):
        return {}, content

    # Find closing ---
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content

    frontmatter_text = content[3 : end_match.start() + 3]
    remaining = content[end_match.end() + 3 :]

    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
        return frontmatter, remaining
    except yaml.YAMLError:
        return {}, content


def extract_sections_markdown(content: str) -> list[tuple[str, str]]:
    """
    Extract sections from markdown content.

    Returns list of (header, section_content) tuples.
    """
    # Split by headers (any level)
    parts = re.split(r"(^#{1,6}\s+.+$)", content, flags=re.MULTILINE)

    sections = []
    current_header = None

    for part in parts:
        if re.match(r"^#{1,6}\s+", part):
            current_header = part.strip().lstrip("#").strip()
        elif part.strip():
            sections.append((current_header, part.strip()))

    return sections


def extract_org_metadata(content: str) -> tuple[dict, str]:
    """
    Extract org-mode metadata from file header.

    Parses #+KEY: value lines at the start of the file.
    Returns (metadata_dict, remaining_content).
    """
    metadata = {}
    lines = content.split("\n")
    body_start = 0

    for i, line in enumerate(lines):
        match = re.match(r"^#\+(\w+):\s*(.*)$", line, re.IGNORECASE)
        if match:
            key = match.group(1).lower()
            value = match.group(2).strip()
            if key in metadata:
                # Handle multiple values (e.g., multiple #+TAGS lines)
                if isinstance(metadata[key], list):
                    metadata[key].append(value)
                else:
                    metadata[key] = [metadata[key], value]
            else:
                metadata[key] = value
            body_start = i + 1
        elif line.strip() and not line.startswith("#"):
            # Stop at first non-metadata, non-comment line
            break

    remaining = "\n".join(lines[body_start:])
    return metadata, remaining


def extract_org_tags(header: str) -> tuple[str, list[str]]:
    """
    Extract tags from an org header line.

    Org tags appear at end of header like: * Header text  :tag1:tag2:
    Returns (header_without_tags, list_of_tags).
    """
    match = re.search(r"\s+(:[:\w]+:)\s*$", header)
    if match:
        tag_str = match.group(1)
        tags = [t for t in tag_str.split(":") if t]
        header_clean = header[: match.start()].strip()
        return header_clean, tags
    return header, []


def extract_sections_org(content: str) -> list[tuple[str, str]]:
    """
    Extract sections from org-mode content.

    Org headers use * (one or more) at start of line.
    Returns list of (header, section_content) tuples.
    """
    # Split by org headers (any level)
    parts = re.split(r"(^\*+\s+.+$)", content, flags=re.MULTILINE)

    sections = []
    current_header = None

    for part in parts:
        if re.match(r"^\*+\s+", part):
            # Remove leading stars and any TODO keywords
            header = re.sub(r"^\*+\s+", "", part)
            # Remove common TODO keywords
            header = re.sub(r"^(TODO|DONE|WAITING|CANCELLED|NEXT|SOMEDAY)\s+", "", header)
            # Extract and remove tags
            header, _ = extract_org_tags(header)
            current_header = header.strip()
        elif part.strip():
            # Skip property drawers
            clean_part = re.sub(r":PROPERTIES:.*?:END:", "", part, flags=re.DOTALL)
            if clean_part.strip():
                sections.append((current_header, clean_part.strip()))

    return sections


def extract_code_context(content: str, file_ext: str) -> dict:
    """
    Extract structural context from code files.

    Returns dict with classes, functions, imports found.
    """
    context = {
        "classes": [],
        "functions": [],
        "imports": [],
    }

    if file_ext == ".py":
        # Python classes and functions
        context["classes"] = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
        context["functions"] = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
        # Top-level imports
        imports = re.findall(r"^(?:from\s+(\S+)|import\s+(\S+))", content, re.MULTILINE)
        context["imports"] = [i[0] or i[1] for i in imports][:10]  # Limit

    elif file_ext in {".js", ".ts", ".jsx", ".tsx"}:
        # JavaScript/TypeScript
        context["classes"] = re.findall(r"class\s+(\w+)", content)
        context["functions"] = re.findall(
            r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
            content,
        )
        context["functions"] = [f[0] or f[1] for f in context["functions"]]
        context["imports"] = re.findall(r"from\s+['\"]([^'\"]+)['\"]", content)[:10]

    return {k: v for k, v in context.items() if v}


def infer_project_from_path(path: Path) -> str | None:
    """
    Infer project name from file path.

    Looks for common patterns like:
    - ~/projects/{project}/...
    - ~/code/{project}/...
    - ~/notes/projects/{project}/...
    """
    parts = path.parts
    project_indicators = {"projects", "code", "repos", "src"}

    for i, part in enumerate(parts):
        if part.lower() in project_indicators and i + 1 < len(parts):
            return parts[i + 1]

    return None


def build_embedding_context(
    chunk_text: str,
    doc_title: str,
    source_path: str,
    source_type: str,
    section_header: str | None = None,
    metadata: DocumentMetadata | None = None,
    code_context: dict | None = None,
) -> str:
    """
    Build contextualized text for embedding.

    This is what the embedding model sees. The original chunk_text
    is stored separately for display.
    """
    parts = []

    # Document identity
    parts.append(f"Document: {doc_title}")

    # Source type context
    if source_type == "code":
        path = Path(source_path)
        parts.append(f"File: {path.name}")
        if code_context:
            if classes := code_context.get("classes"):
                parts.append(f"Classes: {', '.join(classes[:5])}")
            if functions := code_context.get("functions"):
                parts.append(f"Functions: {', '.join(functions[:5])}")

    # Project from metadata or path
    project = None
    if metadata and metadata.project:
        project = metadata.project
    else:
        project = infer_project_from_path(Path(source_path))

    if project:
        parts.append(f"Project: {project}")

    # Section context for long documents
    if section_header:
        parts.append(f"Section: {section_header}")

    # Tags/topics from metadata
    if metadata and metadata.tags:
        parts.append(f"Topics: {', '.join(metadata.tags[:5])}")

    if metadata and metadata.category:
        parts.append(f"Category: {metadata.category}")

    # The actual content
    parts.append(f"Content: {chunk_text}")

    return "\n".join(parts)


def chunk_text(
    text: str,
    chunk_size: int = config.chunk_size,
    chunk_overlap: int = config.chunk_overlap,
) -> Generator[tuple[int, str], None, None]:
    """
    Split text into overlapping chunks.

    Tries to break at paragraph/sentence boundaries.
    Uses approximate token count (4 chars ≈ 1 token).
    """
    char_size = chunk_size * config.chars_per_token
    char_overlap = chunk_overlap * config.chars_per_token

    if len(text) <= char_size:
        yield 0, text
        return

    # Split into paragraphs
    paragraphs = re.split(r"\n\n+", text)

    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= char_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                yield chunk_index, current_chunk.strip()
                chunk_index += 1
                # Keep overlap
                overlap = current_chunk[-char_overlap:] if len(current_chunk) > char_overlap else ""
                current_chunk = overlap + para + "\n\n"
            else:
                # Single paragraph too large - split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= char_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            yield chunk_index, current_chunk.strip()
                            chunk_index += 1
                            overlap = (
                                current_chunk[-char_overlap:]
                                if len(current_chunk) > char_overlap
                                else ""
                            )
                            current_chunk = overlap + sentence + " "
                        else:
                            # Single sentence too large - hard split
                            yield chunk_index, sentence[:char_size]
                            chunk_index += 1
                            current_chunk = (
                                sentence[-char_overlap:] if len(sentence) > char_overlap else ""
                            )

    if current_chunk.strip():
        yield chunk_index, current_chunk.strip()


def parse_markdown(path: Path, extra_metadata: dict | None = None) -> Document:
    """Parse a markdown (.md) file into a Document."""
    content = read_text_with_fallback(path)

    # Extract frontmatter
    frontmatter, body = extract_frontmatter(content)
    metadata = DocumentMetadata.from_frontmatter(frontmatter)

    # Merge extra metadata
    if extra_metadata:
        if "tags" in extra_metadata:
            metadata.tags.extend(extra_metadata["tags"])
        if "project" in extra_metadata:
            metadata.project = extra_metadata["project"]
        if "category" in extra_metadata:
            metadata.category = extra_metadata["category"]

    # Extract title
    title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    title = title_match.group(1) if title_match else path.stem

    # Extract sections
    sections = extract_sections_markdown(body)

    return Document(
        source_path=str(path.resolve()),
        source_type="markdown",
        title=title,
        content=content,
        metadata=metadata,
        sections=sections,
    )


def parse_org(path: Path, extra_metadata: dict | None = None) -> Document:
    """Parse an org-mode (.org) file into a Document."""
    content = read_text_with_fallback(path)

    # Extract org metadata (#+KEY: value lines)
    org_meta, body = extract_org_metadata(content)

    # Build DocumentMetadata from org metadata
    tags = []
    # #+FILETAGS: :tag1:tag2:
    if filetags := org_meta.get("filetags"):
        tags.extend([t for t in filetags.split(":") if t])
    # #+TAGS: tag1 tag2
    if tag_str := org_meta.get("tags"):
        if isinstance(tag_str, list):
            for t in tag_str:
                tags.extend(t.split())
        else:
            tags.extend(tag_str.split())

    metadata = DocumentMetadata(
        tags=tags,
        project=org_meta.get("project"),
        category=org_meta.get("category"),
    )

    # Merge extra metadata
    if extra_metadata:
        if "tags" in extra_metadata:
            metadata.tags.extend(extra_metadata["tags"])
        if "project" in extra_metadata:
            metadata.project = extra_metadata["project"]
        if "category" in extra_metadata:
            metadata.category = extra_metadata["category"]

    # Extract title from #+TITLE or first header
    title = org_meta.get("title")
    if not title:
        title_match = re.search(r"^\*+\s+(.+)$", body, re.MULTILINE)
        if title_match:
            title, _ = extract_org_tags(title_match.group(1))
            # Remove TODO keywords from title
            title = re.sub(r"^(TODO|DONE|WAITING|CANCELLED|NEXT|SOMEDAY)\s+", "", title)
        else:
            title = path.stem

    # Extract sections
    sections = extract_sections_org(body)

    return Document(
        source_path=str(path.resolve()),
        source_type="org",
        title=title,
        content=content,
        metadata=metadata,
        sections=sections,
    )


def parse_text(path: Path, extra_metadata: dict | None = None) -> Document:
    """Parse a plain text file into a Document (no special parsing)."""
    content = read_text_with_fallback(path)

    metadata = DocumentMetadata()
    if extra_metadata:
        metadata = DocumentMetadata(
            tags=extra_metadata.get("tags", []),
            project=extra_metadata.get("project"),
            category=extra_metadata.get("category"),
        )

    return Document(
        source_path=str(path.resolve()),
        source_type="text",
        title=path.stem,
        content=content,
        metadata=metadata,
        sections=[],  # No section parsing for raw text
    )


def parse_code(path: Path, extra_metadata: dict | None = None) -> Document:
    """Parse a code file into a Document."""
    content = read_text_with_fallback(path)

    metadata = DocumentMetadata()
    if extra_metadata:
        metadata = DocumentMetadata(
            tags=extra_metadata.get("tags", []),
            project=extra_metadata.get("project"),
            category=extra_metadata.get("category"),
        )

    # Auto-tag by language
    lang_tags = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".sql": "sql",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    if lang := lang_tags.get(path.suffix):
        if lang not in metadata.tags:
            metadata.tags.append(lang)

    return Document(
        source_path=str(path.resolve()),
        source_type="code",
        title=path.name,
        content=content,
        metadata=metadata,
    )


def is_text_file(path: Path) -> bool:
    """Check if a file appears to be text (not binary)."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        # Check for null bytes (binary indicator)
        if b"\x00" in chunk:
            return False
        # Try to decode as UTF-8
        try:
            chunk.decode("utf-8")
            return True
        except UnicodeDecodeError:
            # Try other common encodings
            for encoding in ("windows-1252", "latin-1"):
                try:
                    chunk.decode(encoding)
                    return True
                except UnicodeDecodeError:
                    continue
            return False
    except OSError:
        return False


def parse_document(path: Path, extra_metadata: dict | None = None, force: bool = False) -> Document:
    """Route document to appropriate parser based on extension.

    If force=True, parse unknown extensions as text/code (for explicitly provided files).
    """
    if path.suffix == ".md":
        return parse_markdown(path, extra_metadata)
    elif path.suffix == ".org":
        return parse_org(path, extra_metadata)
    elif path.suffix in config.document_extensions:
        return parse_text(path, extra_metadata)
    elif path.suffix in config.code_extensions:
        return parse_code(path, extra_metadata)
    elif force:
        # Unknown extension but explicitly requested - treat as code/config file
        return parse_code(path, extra_metadata)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def collect_documents(
    root: Path,
    extra_metadata: dict | None = None,
) -> Generator[Document, None, None]:
    """Recursively collect documents from a directory."""
    print(f"Scanning {root}...", file=sys.stderr, flush=True)
    scanned = 0
    collected = 0
    skipped_dir = 0
    skipped_ext = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        scanned += 1
        if scanned % 500 == 0:
            print(f"  {scanned} files scanned, {collected} documents found...", file=sys.stderr, flush=True)

        if config.should_skip_path(path):
            skipped_dir += 1
            continue

        if path.suffix not in config.all_extensions:
            skipped_ext += 1
            continue

        # Check filename-based skip/block patterns first (before reading content)
        skip_check = check_file_skip(path)
        if skip_check.should_skip:
            prefix = "BLOCKED" if skip_check.is_security else "Skipping"
            print(f"{prefix}: {path} ({skip_check.reason})", file=sys.stderr)
            continue

        try:
            doc = parse_document(path, extra_metadata)

            # Content-based checks (after parsing)
            if config.scan_content:
                skip_check = check_file_skip(path, doc.content)
                if skip_check.should_skip:
                    prefix = "BLOCKED" if skip_check.is_security else "Skipping"
                    print(f"{prefix}: {path} ({skip_check.reason})", file=sys.stderr)
                    continue

            # Capture file mtime for staleness tracking
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            doc.metadata.extra["file_modified_at"] = mtime.isoformat()

            collected += 1
            yield doc
        except Exception as e:
            print(f"Error parsing {path}: {e}", file=sys.stderr)

    if scanned >= 1000:
        print(f"Scan complete: {scanned} files, {skipped_dir} in skipped dirs, {skipped_ext} wrong extension", file=sys.stderr, flush=True)


def create_chunks(doc: Document) -> list[Chunk]:
    """
    Create contextual chunks from a document.

    Each chunk includes:
    - content: original text (for display)
    - embedding_text: contextualized text (for embedding)
    """
    chunks = []

    # For code files, extract structural context once
    code_context = None
    if doc.source_type == "code":
        ext = Path(doc.source_path).suffix
        code_context = extract_code_context(doc.content, ext)

    # Chunk by sections if available, otherwise whole document
    if doc.sections:
        chunk_index = 0
        for section_header, section_content in doc.sections:
            for _, section_chunk in chunk_text_generator(section_content):
                embedding_text = build_embedding_context(
                    chunk_text=section_chunk,
                    doc_title=doc.title,
                    source_path=doc.source_path,
                    source_type=doc.source_type,
                    section_header=section_header,
                    metadata=doc.metadata,
                    code_context=code_context,
                )

                chunks.append(
                    Chunk(
                        content=section_chunk,
                        embedding_text=embedding_text,
                        chunk_index=chunk_index,
                        token_count=len(section_chunk) // config.chars_per_token,
                        metadata={"section": section_header} if section_header else {},
                    )
                )
                chunk_index += 1
    else:
        for chunk_index, chunk_content in chunk_text(doc.content):
            embedding_text = build_embedding_context(
                chunk_text=chunk_content,
                doc_title=doc.title,
                source_path=doc.source_path,
                source_type=doc.source_type,
                metadata=doc.metadata,
                code_context=code_context,
            )

            chunks.append(
                Chunk(
                    content=chunk_content,
                    embedding_text=embedding_text,
                    chunk_index=chunk_index,
                    token_count=len(chunk_content) // config.chars_per_token,
                )
            )

    return chunks


# Alias for the generator to avoid name collision
chunk_text_generator = chunk_text


class Ingester:
    """Handles document ingestion into pgvector."""

    def __init__(self, db_url: str, use_modal: bool = True):
        self.db_url = db_url
        self.use_modal = use_modal
        self._embedder = None

    @property
    def embedder(self):
        """Lazy-load Modal embedder."""
        if self._embedder is None:
            if self.use_modal:
                import modal

                self._embedder = modal.Cls.from_name("knowledge-embedder", "Embedder")()
            else:
                # Fall back to local embedding
                from .local_embedder import embed_document

                class LocalEmbedder:
                    def embed_batch(self, texts):
                        return [embed_document(t) for t in texts]

                self._embedder = LocalEmbedder()
        return self._embedder

    def ingest_documents(self, documents: list[Document], batch_size: int = 50):
        """
        Ingest documents into the database.

        1. Check for existing documents (by hash)
        2. Create contextual chunks
        3. Generate embeddings via Modal (or local)
        4. Store in pgvector
        """
        with psycopg.connect(self.db_url, row_factory=dict_row) as conn:
            register_vector(conn)
            for doc in documents:
                doc_hash = content_hash(doc.content)

                # Check if document exists and unchanged
                existing = conn.execute(
                    "SELECT id FROM documents WHERE content_hash = %s", (doc_hash,)
                ).fetchone()

                if existing:
                    # Content unchanged - but update file_modified_at if present
                    new_mtime = doc.metadata.extra.get("file_modified_at")
                    if new_mtime:
                        conn.execute(
                            """UPDATE documents
                               SET metadata = jsonb_set(metadata, '{file_modified_at}', to_jsonb(%s::text))
                               WHERE id = %s""",
                            (new_mtime, existing["id"]),
                        )
                        conn.commit()
                    print(f"Skipping (unchanged): {doc.source_path}")
                    continue

                # Check if same path exists with different hash (updated file)
                old_doc = conn.execute(
                    "SELECT id FROM documents WHERE source_path = %s", (doc.source_path,)
                ).fetchone()

                if old_doc:
                    print(f"Updating: {doc.source_path}")
                    conn.execute("DELETE FROM documents WHERE id = %s", (old_doc["id"],))
                else:
                    print(f"Ingesting: {doc.source_path}")

                # Insert document (ON CONFLICT handles duplicate content from different paths)
                result = conn.execute(
                    """
                    INSERT INTO documents (source_path, source_type, title, content, metadata, content_hash)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO NOTHING
                    RETURNING id
                    """,
                    (
                        doc.source_path,
                        doc.source_type,
                        doc.title,
                        doc.content,
                        psycopg.types.json.Json(doc.metadata.to_dict()),
                        doc_hash,
                    ),
                ).fetchone()

                if result is None:
                    print(f"  Skipping (duplicate content): {doc.source_path}")
                    continue

                doc_id = result["id"]

                # Create chunks
                chunks = create_chunks(doc)

                if not chunks:
                    conn.commit()
                    continue

                # Generate embeddings (batch to avoid OOM on GPU)
                embedding_texts = [c.embedding_text for c in chunks]
                embed_batch_size = 100  # Max texts per GPU call

                print(f"  Generating embeddings for {len(chunks)} chunks...")
                if self.use_modal:
                    embeddings = []
                    for i in range(0, len(embedding_texts), embed_batch_size):
                        batch = embedding_texts[i : i + embed_batch_size]
                        embeddings.extend(self.embedder.embed_batch.remote(batch))
                else:
                    embeddings = self.embedder.embed_batch(embedding_texts)

                # Insert chunks with embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    conn.execute(
                        """
                        INSERT INTO chunks 
                            (document_id, chunk_index, content, embedding_text, embedding, token_count, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            doc_id,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.embedding_text,
                            embedding,
                            chunk.token_count,
                            psycopg.types.json.Json(chunk.metadata),
                        ),
                    )

                conn.commit()
                print(f"  → {len(chunks)} chunks indexed")

    def delete_document(self, source_path: str):
        """Remove a document and its chunks."""
        with psycopg.connect(self.db_url) as conn:
            result = conn.execute(
                "DELETE FROM documents WHERE source_path = %s RETURNING id",
                (source_path,),
            ).fetchone()
            conn.commit()
            return result is not None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ingest.py ~/notes
    python ingest.py ~/projects/myapp --metadata '{"project": "myapp"}'
    python ingest.py document.md --local  # Use CPU embedding
        """,
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to ingest")
    parser.add_argument(
        "--metadata",
        type=json.loads,
        default={},
        help='JSON metadata to attach (e.g., \'{"project": "myapp"}\')',
    )
    parser.add_argument(
        "--db-url",
        default=config.db_url,
        help="Database URL",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local CPU embedding instead of Modal",
    )

    args = parser.parse_args()

    ingester = Ingester(args.db_url, use_modal=not args.local)

    # Collect documents
    documents = []
    for path in args.paths:
        path = path.resolve()
        if path.is_dir():
            documents.extend(collect_documents(path, args.metadata))
        elif path.is_file():
            # Check security patterns first
            skip_check = check_file_skip(path)
            if skip_check.should_skip:
                prefix = "BLOCKED" if skip_check.is_security else "Skipping"
                print(f"{prefix}: {path} ({skip_check.reason})", file=sys.stderr)
                continue

            # For explicitly provided files, try to parse even with unknown extension
            if path.suffix in config.all_extensions:
                documents.append(parse_document(path, args.metadata))
            elif is_text_file(path):
                print(f"Parsing as text: {path}", file=sys.stderr)
                documents.append(parse_document(path, args.metadata, force=True))
            else:
                print(f"Skipping binary file: {path}", file=sys.stderr)

    if not documents:
        print("No documents found to ingest")
        return

    print(f"Found {len(documents)} documents to process")
    ingester.ingest_documents(documents)
    print("Done!")


if __name__ == "__main__":
    main()
