"""Database-level synthesis - generate synthetic knowledge documents from broad analysis."""

from __future__ import annotations

import hashlib
import json
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row


@dataclass
class SynthesisResult:
    """Result from a synthesis run."""

    proposals_created: int
    model: str
    origin: str
    project: str | None = None
    proposals: list[dict] = field(default_factory=list)
    raw_response: str | None = None


SYNTHESIS_SYSTEM = """You are analyzing a personal knowledge base to synthesize useful reference \
documents. Based on the provided statistics and document samples, propose multiple synthetic \
knowledge documents that would be valuable additions.

Types of documents to propose:
- Topic summaries: consolidate scattered information about a subject
- Entity profiles: synthesize what's known about a person, project, or technology
- Relationship maps: describe how concepts/projects/people connect
- Cross-cutting insights: patterns or themes that span multiple documents

Each proposal should contain genuinely useful, specific content derived from the samples — \
not generic placeholders.

Return ONLY valid JSON: an array of objects with "title" and "content" fields.
Example: [{"title": "Overview of ...", "content": "Detailed synthesis..."}]"""

SYNTHESIS_USER = """## Content Statistics
{stats}

## Sample Document Titles and Excerpts ({sample_count} documents)
{document_samples}

---
Propose {min_proposals}-{max_proposals} synthetic knowledge documents as a JSON array.
Each should have a "title" and "content" field.
Content should be substantive (3-10 paragraphs) and synthesize information from the samples."""


def synthesize(
    db_url: str,
    project: str | None = None,
    sample_size: int = 25,
    max_proposals: int = 10,
    origin: str = "cli",
) -> SynthesisResult:
    """Sample the database and propose synthetic knowledge documents.

    Args:
        db_url: Database URL
        project: Filter by project
        sample_size: Number of documents to sample
        max_proposals: Maximum proposals to generate
        origin: Origin identifier ('cli' or 'mcp')

    Returns:
        SynthesisResult with created proposals
    """
    from . import get_llm
    from .analyze import get_content_stats, get_document_samples

    llm = get_llm()
    if llm is None:
        raise RuntimeError(
            "No LLM provider configured. Synthesis requires an LLM. "
            "Set ANTHROPIC_API_KEY or configure llm.provider in config."
        )

    # Gather data
    stats = get_content_stats(db_url, project)
    samples = get_document_samples(db_url, project, sample_size, strategy="diverse")

    # Format stats
    stats_text = []
    stats_text.append(f"Total documents: {stats['total_documents']}")
    stats_text.append(f"Total tokens: ~{stats['total_tokens']:,}")
    if stats["source_types"]:
        types_str = ", ".join(
            f"{t}: {c}" for t, c in sorted(stats["source_types"].items(), key=lambda x: -x[1])
        )
        stats_text.append(f"Source types: {types_str}")
    if stats["projects"]:
        stats_text.append(f"Projects: {', '.join(stats['projects'])}")
    if stats["date_range"]["earliest"]:
        stats_text.append(
            f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}"
        )

    # Format samples
    sample_text = []
    for s in samples:
        excerpt = s["excerpt"].replace("\n", " ")[:200] if s["excerpt"] else ""
        sample_text.append(f"### {s['title']} ({s['source_type']})\n{excerpt}...")

    min_proposals = max(3, max_proposals // 2)
    prompt = SYNTHESIS_USER.format(
        stats="\n".join(stats_text),
        sample_count=len(samples),
        document_samples="\n\n".join(sample_text),
        min_proposals=min_proposals,
        max_proposals=max_proposals,
    )

    response = llm.complete(prompt, system=SYNTHESIS_SYSTEM, max_tokens=4096, timeout=120)

    # Parse proposals from response
    proposals = _parse_proposals(response.content)

    if not proposals:
        return SynthesisResult(
            proposals_created=0,
            model=response.model,
            origin=origin,
            project=project,
            raw_response=response.content,
        )

    # Insert into pending_synthesis
    created = []
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        for p in proposals:
            row = conn.execute(
                """
                INSERT INTO pending_synthesis (title, content, project, model, origin, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, title
                """,
                (
                    p["title"],
                    p["content"],
                    project,
                    response.model,
                    origin,
                    psycopg.types.json.Json({}),
                ),
            ).fetchone()
            created.append({"id": str(row["id"]), "title": row["title"]})
        conn.commit()

    return SynthesisResult(
        proposals_created=len(created),
        model=response.model,
        origin=origin,
        project=project,
        proposals=created,
    )


def _parse_proposals(content: str) -> list[dict]:
    """Parse LLM response into list of {title, content} dicts."""
    text = content.strip()

    # Strip markdown code blocks (```json ... ``` or ``` ... ```)
    import re

    code_block = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    # Try direct parse first
    for candidate in [text, _extract_json_array(text)]:
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return [
                    {"title": item["title"], "content": item["content"]}
                    for item in data
                    if isinstance(item, dict) and "title" in item and "content" in item
                ]
        except (json.JSONDecodeError, KeyError):
            continue

    return []


def _extract_json_array(text: str) -> str | None:
    """Find the outermost JSON array in text, handling nested brackets."""
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def list_pending_synthesis(
    db_url: str,
    project: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List pending synthesis proposals."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        sql = """
            SELECT id, title, LEFT(content, 200) as excerpt,
                   project, model, origin, created_at
            FROM pending_synthesis
            WHERE status = 'pending'
        """
        params: list[Any] = []
        if project:
            sql += " AND project = %s"
            params.append(project)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_pending_synthesis(db_url: str, pending_id: str) -> dict | None:
    """Get a single pending synthesis proposal."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        row = conn.execute(
            """
            SELECT id, title, content, project, model, origin, status,
                   metadata, created_at, reviewed_at
            FROM pending_synthesis
            WHERE id = %s
            """,
            (pending_id,),
        ).fetchone()
        return dict(row) if row else None


def approve_synthesis(
    db_url: str,
    pending_id: str,
    title: str | None = None,
    content: str | None = None,
) -> str | None:
    """Approve a pending synthesis, creating a searchable document.

    Args:
        db_url: Database URL
        pending_id: ID of the pending synthesis
        title: Optional title override
        content: Optional content override

    Returns:
        source_path of created document, or None on failure
    """
    from ..local_embedder import embed_document

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        row = conn.execute(
            "SELECT * FROM pending_synthesis WHERE id = %s AND status = 'pending'",
            (pending_id,),
        ).fetchone()

        if not row:
            return None

        final_title = title or row["title"]
        final_content = content or row["content"]

        # Generate source path
        synthesis_id = str(uuid.uuid4())
        source_path = f"okb://synthesis/{synthesis_id}"

        # Build metadata
        metadata = {
            "model": row["model"],
            "origin": row["origin"],
            "synthesized_at": datetime.now(UTC).isoformat(),
            "pending_id": str(row["id"]),
        }
        if row["project"]:
            metadata["project"] = row["project"]

        # Content hash
        content_hash = hashlib.sha256(final_content.encode()).hexdigest()[:16]

        # Build embedding text
        embedding_parts = [f"Document: {final_title}"]
        if row["project"]:
            embedding_parts.append(f"Project: {row['project']}")
        embedding_parts.append(f"Content: {final_content}")
        embedding_text = "\n".join(embedding_parts)

        # Generate embedding
        embedding = embed_document(embedding_text)

        # Insert document
        doc_id = conn.execute(
            """
            INSERT INTO documents (source_path, source_type, title, content,
                                   metadata, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                source_path,
                "synthesis",
                final_title,
                final_content,
                psycopg.types.json.Json(metadata),
                content_hash,
            ),
        ).fetchone()["id"]

        # Insert single chunk
        token_count = len(final_content) // 4
        conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content,
                               embedding_text, embedding, token_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc_id,
                0,
                final_content,
                embedding_text,
                embedding,
                token_count,
                psycopg.types.json.Json({}),
            ),
        )

        # Mark pending as approved
        conn.execute(
            """
            UPDATE pending_synthesis SET status = 'approved', reviewed_at = NOW()
            WHERE id = %s
            """,
            (pending_id,),
        )

        conn.commit()
        return source_path


# Thread pool for async approvals
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2)
    return _executor


def approve_synthesis_async(
    db_url: str,
    pending_id: str,
    title: str | None = None,
    content: str | None = None,
) -> Future:
    """Submit async approval (embedding is slow)."""
    return _get_executor().submit(approve_synthesis, db_url, pending_id, title, content)


def shutdown_executor(wait: bool = True) -> None:
    """Shutdown the thread pool executor."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None


def reject_synthesis(db_url: str, pending_id: str) -> bool:
    """Reject a pending synthesis proposal."""
    with psycopg.connect(db_url) as conn:
        result = conn.execute(
            """
            UPDATE pending_synthesis SET status = 'rejected', reviewed_at = NOW()
            WHERE id = %s AND status = 'pending'
            """,
            (pending_id,),
        )
        conn.commit()
        return result.rowcount > 0


def update_pending_synthesis(
    db_url: str,
    pending_id: str,
    title: str | None = None,
    content: str | None = None,
) -> bool:
    """Edit a pending synthesis proposal before approve/reject."""
    updates = []
    params: list[Any] = []

    if title is not None:
        updates.append("title = %s")
        params.append(title)
    if content is not None:
        updates.append("content = %s")
        params.append(content)

    if not updates:
        return False

    params.append(pending_id)
    with psycopg.connect(db_url) as conn:
        result = conn.execute(
            f"""
            UPDATE pending_synthesis SET {', '.join(updates)}
            WHERE id = %s AND status = 'pending'
            """,
            params,
        )
        conn.commit()
        return result.rowcount > 0
