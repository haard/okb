"""
MCP Server for Knowledge Base.

Exposes semantic search to Claude Code via the Model Context Protocol.

Usage:
    okb serve

Configure in Claude Code (see https://docs.anthropic.com/en/docs/claude-code):
    {
      "mcpServers": {
        "knowledge-base": {
          "command": "okb",
          "args": ["serve"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

import dateparser
import psycopg
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    Tool,
)
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from . import __version__ as _okb_version
from .config import config
from .local_embedder import embed_document, embed_query, warmup
from .tools import format_actionable_items as format_actionable_items  # noqa: F401 re-export
from .tools import format_relative_time
from .tools import format_search_results as format_search_results  # noqa: F401 re-export
from .tools import get_document_date as get_document_date  # noqa: F401 re-export


def _dateparser_parse(text: str) -> datetime | None:
    """Parse a natural-language date string via dateparser (returns UTC-aware datetime or None)."""
    return dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "past",
            "TIMEZONE": "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
        },
    )


def parse_since_filter(since: str) -> datetime | None:
    """Parse since filter: compact ('7d','6mo','1y'), ISO date, or natural language."""
    from datetime import timedelta

    now = datetime.now(UTC)
    # Fast-path: compact relative format
    match = re.match(r"^(\d+)(d|mo|y)$", since.lower())
    if match:
        value, unit = int(match.group(1)), match.group(2)
        days = value * {"d": 1, "mo": 30, "y": 365}[unit]
        return now - timedelta(days=days)
    # Fast-path: ISO date
    try:
        return datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError:
        pass
    # Fallback: dateparser for natural language
    return _dateparser_parse(since)


def parse_date_range(date_str: str) -> tuple[datetime, datetime] | None:
    """Parse date range: keywords, YYYY-MM-DD, or natural language (single-day range)."""
    from datetime import timedelta

    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    keyword = date_str.lower().replace(" ", "_")
    if keyword == "today":
        return (today_start, today_end)
    elif keyword == "tomorrow":
        return (today_end, today_end + timedelta(days=1))
    elif keyword == "yesterday":
        return (today_start - timedelta(days=1), today_start)
    elif keyword == "this_week":
        days_since_monday = now.weekday()
        week_start = today_start - timedelta(days=days_since_monday)
        return (week_start, week_start + timedelta(days=7))
    elif keyword == "next_week":
        days_since_monday = now.weekday()
        next_week_start = today_start + timedelta(days=7 - days_since_monday)
        return (next_week_start, next_week_start + timedelta(days=7))
    elif keyword == "last_week":
        days_since_monday = now.weekday()
        this_week_start = today_start - timedelta(days=days_since_monday)
        return (this_week_start - timedelta(days=7), this_week_start)
    elif keyword == "this_month":
        month_start = today_start.replace(day=1)
        if now.month == 12:
            month_end = month_start.replace(year=now.year + 1, month=1)
        else:
            month_end = month_start.replace(month=now.month + 1)
        return (month_start, month_end)
    elif keyword == "next_month":
        if now.month == 12:
            next_start = today_start.replace(year=now.year + 1, month=1, day=1)
        else:
            next_start = today_start.replace(month=now.month + 1, day=1)
        if next_start.month == 12:
            next_end = next_start.replace(year=next_start.year + 1, month=1)
        else:
            next_end = next_start.replace(month=next_start.month + 1)
        return (next_start, next_end)

    # YYYY-MM-DD fast-path
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        try:
            dt = datetime.fromisoformat(date_str).replace(tzinfo=UTC)
            return (dt, dt + timedelta(days=1))
        except ValueError:
            return None

    # Fallback: dateparser → single-day range
    parsed = _dateparser_parse(date_str)
    if parsed:
        day_start = parsed.replace(hour=0, minute=0, second=0, microsecond=0)
        return (day_start, day_start + timedelta(days=1))
    return None


class KnowledgeBase:
    """Knowledge base with semantic and keyword search."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._conn = None

    def get_connection(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.db_url, row_factory=dict_row)
            register_vector(self._conn)
        return self._conn

    def close(self):
        """Close the database connection if open."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        source_type: str | None = None,
        project: str | None = None,
        min_score: float = 0.25,
        since: str | None = None,
    ) -> list[dict]:
        """
        Search for semantically similar chunks.

        Returns chunks with their parent document context.
        """
        embedding = embed_query(query)
        conn = self.get_connection()

        # Build query with optional filters
        sql = """
            SELECT
                c.content,
                c.chunk_index,
                c.metadata as chunk_metadata,
                d.source_path,
                d.source_type,
                d.title,
                d.metadata as doc_metadata,
                1 - (c.embedding <=> %s::vector) as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE 1 - (c.embedding <=> %s::vector) > %s
        """
        params: list[Any] = [embedding, embedding, min_score]

        if source_type:
            sql += " AND d.source_type = %s"
            params.append(source_type)

        if project:
            sql += " AND d.metadata->>'project' = %s"
            params.append(project)

        if since:
            since_dt = parse_since_filter(since)
            if since_dt:
                sql += """ AND COALESCE(
                    (d.metadata->>'document_date')::timestamptz,
                    (d.metadata->>'file_modified_at')::timestamptz
                ) >= %s"""
                params.append(since_dt)

        sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
        params.extend([embedding, min(limit, config.max_limit)])

        results = conn.execute(sql, params).fetchall()
        return [dict(r) for r in results]

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        source_type: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        """
        Full-text keyword search.

        Better for exact matches, code symbols, function names.
        """
        conn = self.get_connection()

        sql = """
            SELECT
                c.content,
                c.chunk_index,
                d.source_path,
                d.source_type,
                d.title,
                d.metadata as doc_metadata,
                ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', %s)) as rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
        """
        params: list[Any] = [query, query]

        if source_type:
            sql += " AND d.source_type = %s"
            params.append(source_type)

        if since:
            since_dt = parse_since_filter(since)
            if since_dt:
                sql += """ AND COALESCE(
                    (d.metadata->>'document_date')::timestamptz,
                    (d.metadata->>'file_modified_at')::timestamptz
                ) >= %s"""
                params.append(since_dt)

        sql += " ORDER BY rank DESC LIMIT %s"
        params.append(min(limit, config.max_limit))

        results = conn.execute(sql, params).fetchall()
        return [dict(r) for r in results]

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        source_type: str | None = None,
        semantic_weight: float = 0.7,
        since: str | None = None,
    ) -> list[dict]:
        """
        Hybrid search combining semantic and keyword results.

        Uses Reciprocal Rank Fusion (RRF) to merge results.
        """
        # Get both result sets
        semantic_results = self.semantic_search(
            query, limit=limit * 2, source_type=source_type, since=since
        )
        keyword_results = self.keyword_search(
            query, limit=limit * 2, source_type=source_type, since=since
        )

        # RRF scoring
        k = 60  # RRF constant
        scores: dict[str, float] = {}
        results_map: dict[str, dict] = {}

        for rank, r in enumerate(semantic_results):
            key = f"{r['source_path']}:{r['chunk_index']}"
            scores[key] = scores.get(key, 0) + semantic_weight / (k + rank + 1)
            results_map[key] = r

        for rank, r in enumerate(keyword_results):
            key = f"{r['source_path']}:{r['chunk_index']}"
            scores[key] = scores.get(key, 0) + (1 - semantic_weight) / (k + rank + 1)
            if key not in results_map:
                results_map[key] = r

        # Sort by combined score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        return [results_map[k] for k in sorted_keys[:limit]]

    def list_sources(self) -> list[dict]:
        """List all indexed sources with stats."""
        conn = self.get_connection()
        results = conn.execute("SELECT * FROM index_stats").fetchall()
        return [dict(r) for r in results]

    def list_projects(self) -> list[str]:
        """List all known projects."""
        conn = self.get_connection()
        results = conn.execute("""
            SELECT DISTINCT metadata->>'project' as project
            FROM documents
            WHERE metadata->>'project' IS NOT NULL
            ORDER BY project
        """).fetchall()
        return [r["project"] for r in results]

    def get_project_stats(self) -> list[dict]:
        """Get projects with document counts for consolidation review."""
        conn = self.get_connection()
        results = conn.execute("""
            SELECT
                metadata->>'project' as project,
                COUNT(*) as doc_count,
                array_agg(DISTINCT source_type) as source_types
            FROM documents
            WHERE metadata->>'project' IS NOT NULL
            GROUP BY metadata->>'project'
            ORDER BY doc_count DESC, project
        """).fetchall()
        return [dict(r) for r in results]

    def list_documents_by_project(self, project: str, limit: int = 100) -> list[dict]:
        """List documents for a specific project."""
        conn = self.get_connection()
        rows = conn.execute(
            """SELECT source_path, title, source_type FROM documents
               WHERE metadata->>'project' = %s ORDER BY title LIMIT %s""",
            (project, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def rename_project(self, old_name: str, new_name: str) -> int:
        """Rename a project (update all documents). Returns count of updated docs."""
        conn = self.get_connection()
        result = conn.execute(
            """
            UPDATE documents
            SET metadata = jsonb_set(metadata, '{project}', %s::jsonb)
            WHERE metadata->>'project' = %s
            """,
            (f'"{new_name}"', old_name),
        )
        conn.commit()
        return result.rowcount

    def set_document_project(self, source_path: str, project: str | None) -> bool:
        """Set or clear the project for a single document."""
        conn = self.get_connection()
        if project:
            result = conn.execute(
                """
                UPDATE documents
                SET metadata = jsonb_set(metadata, '{project}', %s::jsonb)
                WHERE source_path = %s
                """,
                (f'"{project}"', source_path),
            )
        else:
            result = conn.execute(
                """
                UPDATE documents
                SET metadata = metadata - 'project'
                WHERE source_path = %s
                """,
                (source_path,),
            )
        conn.commit()
        return result.rowcount > 0

    def get_document(self, source_path: str) -> dict | None:
        """Get full document content by path."""
        conn = self.get_connection()
        result = conn.execute(
            "SELECT * FROM documents WHERE source_path = %s", (source_path,)
        ).fetchone()
        return dict(result) if result else None

    def get_recent_documents(self, limit: int = 10) -> list[dict]:
        """Get recently indexed documents."""
        conn = self.get_connection()
        results = conn.execute(
            """
            SELECT source_path, source_type, title, metadata, updated_at
            FROM documents
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in results]

    def save_knowledge(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        project: str | None = None,
        source_type: str = "claude-note",
    ) -> dict:
        """
        Save a piece of knowledge directly from Claude.

        Creates a virtual document (not file-backed) with embedding.
        Args:
            source_type: 'claude-note' (default) or 'synthesis'
        Returns the saved document info.
        """
        conn = self.get_connection()

        # Generate unique source path based on type
        knowledge_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        if source_type == "synthesis":
            source_path = f"okb://synthesis/{timestamp}-{knowledge_id}"
        else:
            source_path = f"claude://knowledge/{timestamp}-{knowledge_id}"
            source_type = "claude-note"

        # Build metadata
        metadata = {}
        if tags:
            metadata["tags"] = tags
        if project:
            metadata["project"] = project
        metadata["source"] = "claude"

        # Content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Check for duplicate content
        existing = conn.execute(
            "SELECT source_path, title FROM documents WHERE content_hash = %s",
            (content_hash,),
        ).fetchone()
        if existing:
            return {
                "status": "duplicate",
                "existing_path": existing["source_path"],
                "existing_title": existing["title"],
            }

        # Build contextual embedding text
        embedding_parts = [f"Document: {title}"]
        if project:
            embedding_parts.append(f"Project: {project}")
        if tags:
            embedding_parts.append(f"Topics: {', '.join(tags)}")
        embedding_parts.append(f"Content: {content}")
        embedding_text = "\n".join(embedding_parts)

        # Generate embedding
        embedding = embed_document(embedding_text)

        # Insert document
        doc_id = conn.execute(
            """
            INSERT INTO documents (source_path, source_type, title, content, metadata, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                source_path,
                source_type,
                title,
                content,
                psycopg.types.json.Json(metadata),
                content_hash,
            ),
        ).fetchone()["id"]

        # Insert single chunk
        token_count = len(content) // 4  # Approximate
        conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content, embedding_text, embedding, token_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc_id,
                0,
                content,
                embedding_text,
                embedding,
                token_count,
                psycopg.types.json.Json({}),
            ),
        )

        conn.commit()

        return {
            "status": "saved",
            "source_path": source_path,
            "title": title,
            "token_count": token_count,
        }

    def update_knowledge(
        self,
        source_path: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
    ) -> dict:
        """Update an existing knowledge document, preserving source_path and created_at."""
        conn = self.get_connection()

        # Fetch existing document
        doc = conn.execute(
            "SELECT id, title, content, metadata, content_hash FROM documents WHERE source_path = %s",
            (source_path,),
        ).fetchone()
        if not doc:
            return {"status": "not_found", "source_path": source_path}

        # Merge: use provided values, fall back to existing
        existing_meta = doc["metadata"] or {}
        new_title = title if title is not None else doc["title"]
        new_content = content if content is not None else doc["content"]
        new_tags = tags if tags is not None else existing_meta.get("tags")
        new_project = project if project is not None else existing_meta.get("project")

        # Build metadata
        new_meta = dict(existing_meta)
        if new_tags is not None:
            new_meta["tags"] = new_tags
        elif "tags" in new_meta and tags is not None:
            del new_meta["tags"]
        if new_project is not None:
            new_meta["project"] = new_project
        elif "project" in new_meta and project is not None:
            del new_meta["project"]

        # Content hash for deduplication
        new_content_hash = hashlib.sha256(new_content.encode()).hexdigest()[:16]

        # Check for duplicate content (only if content changed, and not self)
        if new_content_hash != doc["content_hash"]:
            existing = conn.execute(
                "SELECT source_path, title FROM documents WHERE content_hash = %s AND source_path != %s",
                (new_content_hash, source_path),
            ).fetchone()
            if existing:
                return {
                    "status": "duplicate",
                    "existing_path": existing["source_path"],
                    "existing_title": existing["title"],
                }

        # Build contextual embedding text
        embedding_parts = [f"Document: {new_title}"]
        if new_project:
            embedding_parts.append(f"Project: {new_project}")
        if new_tags:
            embedding_parts.append(f"Topics: {', '.join(new_tags)}")
        embedding_parts.append(f"Content: {new_content}")
        embedding_text = "\n".join(embedding_parts)

        # Generate embedding
        embedding = embed_document(embedding_text)

        # Update document
        conn.execute(
            """
            UPDATE documents
            SET title = %s, content = %s, metadata = %s, content_hash = %s, updated_at = NOW()
            WHERE source_path = %s
            """,
            (
                new_title,
                new_content,
                psycopg.types.json.Json(new_meta),
                new_content_hash,
                source_path,
            ),
        )

        # Replace chunks
        doc_id = doc["id"]
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        token_count = len(new_content) // 4
        conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content, embedding_text, embedding, token_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (doc_id, 0, new_content, embedding_text, embedding, token_count, psycopg.types.json.Json({})),
        )

        conn.commit()

        return {
            "status": "updated",
            "source_path": source_path,
            "title": new_title,
            "token_count": token_count,
        }

    def delete_knowledge(self, source_path: str) -> bool:
        """Delete a document by source path."""
        conn = self.get_connection()
        result = conn.execute(
            "DELETE FROM documents WHERE source_path = %s RETURNING id",
            (source_path,),
        ).fetchone()
        conn.commit()
        return result is not None

    def save_todo(
        self,
        title: str,
        content: str | None = None,
        due_date: str | None = None,
        priority: str | None = None,
        project: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """
        Create a TODO item in the knowledge base.

        Args:
            title: TODO item title
            content: Optional description/notes
            due_date: Due date (ISO date or 'today'/'tomorrow')
            priority: Priority ('A'/'B'/'C' or 1-5, 1=highest)
            project: Project name
            tags: List of tags

        Returns:
            Dict with status and saved document info
        """
        conn = self.get_connection()

        # Generate unique source path
        todo_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        source_path = f"claude://todo/{timestamp}-{todo_id}"

        # Parse priority: A=1, B=2, C=3, or numeric 1-5
        parsed_priority = None
        if priority:
            priority_map = {"A": 1, "B": 2, "C": 3, "a": 1, "b": 2, "c": 3}
            if priority.upper() in priority_map:
                parsed_priority = priority_map[priority.upper()]
            elif priority.isdigit() and 1 <= int(priority) <= 5:
                parsed_priority = int(priority)

        # Parse due_date
        parsed_due_date = None
        if due_date:
            date_range = parse_date_range(due_date)
            if date_range:
                parsed_due_date = date_range[0]  # Use start of range
            else:
                # Try ISO format
                try:
                    parsed_due_date = datetime.fromisoformat(due_date.replace("Z", "+00:00"))
                except ValueError:
                    pass

        # Build metadata
        metadata = {"source": "claude"}
        if tags:
            metadata["tags"] = tags
        if project:
            metadata["project"] = project

        # Use content if provided, otherwise use title
        doc_content = content if content else title

        # Content hash for deduplication
        content_hash = hashlib.sha256(f"{title}:{doc_content}".encode()).hexdigest()[:16]

        # Build contextual embedding text
        embedding_parts = [f"TODO: {title}"]
        if project:
            embedding_parts.append(f"Project: {project}")
        if tags:
            embedding_parts.append(f"Topics: {', '.join(tags)}")
        if content:
            embedding_parts.append(f"Details: {content}")
        embedding_text = "\n".join(embedding_parts)

        # Generate embedding
        embedding = embed_document(embedding_text)

        # Insert document with structured fields
        doc_id = conn.execute(
            """
            INSERT INTO documents (
                source_path, source_type, title, content, metadata, content_hash,
                status, priority, due_date
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                source_path,
                "claude-todo",
                title,
                doc_content,
                psycopg.types.json.Json(metadata),
                content_hash,
                "pending",
                parsed_priority,
                parsed_due_date,
            ),
        ).fetchone()["id"]

        # Insert single chunk
        token_count = len(doc_content) // 4  # Approximate
        conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content, embedding_text, embedding, token_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc_id,
                0,
                doc_content,
                embedding_text,
                embedding,
                token_count,
                psycopg.types.json.Json({}),
            ),
        )

        conn.commit()

        return {
            "status": "saved",
            "source_path": source_path,
            "title": title,
            "priority": parsed_priority,
            "due_date": str(parsed_due_date) if parsed_due_date else None,
        }

    def get_database_metadata(self) -> dict:
        """Get LLM-enhanced database metadata."""
        conn = self.get_connection()
        results = conn.execute("SELECT key, value, source FROM database_metadata").fetchall()
        return {r["key"]: {"value": r["value"], "source": r["source"]} for r in results}

    def set_database_metadata(self, key: str, value: Any) -> bool:
        """Set or update LLM-enhanced database metadata."""
        conn = self.get_connection()
        conn.execute(
            """
            INSERT INTO database_metadata (key, value, source, updated_at)
            VALUES (%s, %s, 'llm', NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                source = 'llm',
                updated_at = NOW()
            """,
            (key, psycopg.types.json.Json(value)),
        )
        conn.commit()
        return True

    def get_actionable_items(
        self,
        item_type: str | None = None,
        status: str | None = None,
        due_date: str | None = None,
        event_date: str | None = None,
        min_priority: int | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Query actionable items (tasks, events, emails) with structured filters.

        Args:
            item_type: Filter by source_type (e.g., 'todoist-task', 'gcal-event', 'gmail')
            status: Filter by status ('pending', 'completed', etc.)
            due_date: Filter tasks due on date ('today', 'tomorrow', 'this_week', 'YYYY-MM-DD')
            event_date: Filter events on date ('today', 'tomorrow', 'this_week', 'YYYY-MM-DD')
            min_priority: Filter items with priority <= this value (1=highest)
            limit: Max results to return
        """
        conn = self.get_connection()

        sql = """
            SELECT
                d.source_path,
                d.source_type,
                d.title,
                d.content,
                d.metadata,
                d.due_date,
                d.event_start,
                d.event_end,
                d.status,
                d.priority
            FROM documents d
            WHERE 1=1
        """
        params: list[Any] = []

        if item_type:
            sql += " AND d.source_type = %s"
            params.append(item_type)

        if status:
            sql += " AND d.status = %s"
            params.append(status)

        if due_date:
            date_range = parse_date_range(due_date)
            if date_range:
                sql += " AND d.due_date >= %s AND d.due_date < %s"
                params.extend(date_range)

        if event_date:
            date_range = parse_date_range(event_date)
            if date_range:
                # Event overlaps with date range
                sql += " AND d.event_start < %s AND (d.event_end > %s OR d.event_end IS NULL)"
                params.extend([date_range[1], date_range[0]])

        if min_priority is not None:
            sql += " AND d.priority IS NOT NULL AND d.priority <= %s"
            params.append(min_priority)

        # Order by: due_date/event_start (soonest first), then priority
        sql += """
            ORDER BY
                COALESCE(d.due_date, d.event_start) ASC NULLS LAST,
                d.priority ASC NULLS LAST
            LIMIT %s
        """
        params.append(min(limit, config.max_limit))

        results = conn.execute(sql, params).fetchall()
        return [dict(r) for r in results]


def _get_sync_state(conn, source_name: str, db_name: str):
    """Get sync state from database."""
    from .plugins.base import SyncState

    result = conn.execute(
        """SELECT last_sync, cursor, extra FROM sync_state
           WHERE source_name = %s AND database_name = %s""",
        (source_name, db_name),
    ).fetchone()

    if result:
        return SyncState(
            last_sync=result["last_sync"],
            cursor=result["cursor"],
            extra=result["extra"] or {},
        )
    return None


def _save_sync_state(conn, source_name: str, db_name: str, state):
    """Save sync state to database."""
    import json

    conn.execute(
        """INSERT INTO sync_state (source_name, database_name, last_sync, cursor, extra, updated_at)
           VALUES (%s, %s, %s, %s, %s, NOW())
           ON CONFLICT (source_name, database_name)
           DO UPDATE SET last_sync = EXCLUDED.last_sync,
                        cursor = EXCLUDED.cursor,
                        extra = EXCLUDED.extra,
                        updated_at = NOW()""",
        (source_name, db_name, state.last_sync, state.cursor, json.dumps(state.extra)),
    )
    conn.commit()


def _run_sync(
    db_url: str,
    sources: list[str],
    sync_all: bool = False,
    full: bool = False,
    doc_ids: list[str] | None = None,
    repos: list[str] | None = None,
    branch: str | None = None,
    db_name: str | None = None,
    include_issues: bool = False,
    include_prs: bool = False,
    include_wiki: bool = False,
    include_source: bool = False,
    folders: list[str] | None = None,
    channels: list[str] | None = None,
) -> str:
    """Run sync for specified sources and return formatted result."""
    from psycopg.rows import dict_row

    from .ingest import Ingester
    from .plugins.registry import PluginRegistry

    # Resolve db_name for source config lookup and sync state
    if db_name is None:
        db_name = config.get_database().name

    # Determine which sources to sync
    if sync_all:
        source_names = config.list_enabled_sources(db_name)
    elif sources:
        source_names = list(sources)
    else:
        # Return list of available sources
        installed = PluginRegistry.list_sources()
        configured = config.list_enabled_sources(db_name)
        lines = ["Available API sources:"]
        for name in installed:
            status = "enabled" if name in configured else "disabled"
            lines.append(f"  - {name} ({status})")
        if not installed:
            lines.append("  (none installed)")
        return "\n".join(lines)

    if not source_names:
        return "No sources to sync."

    results = []
    ingester = Ingester(db_url, use_modal=True)

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        for source_name in source_names:
            # Get the plugin
            source = PluginRegistry.get_source(source_name)
            if source is None:
                results.append(f"{source_name}: not found")
                continue

            # Get and resolve config (per-db sources override global)
            source_cfg = config.get_source_config(source_name, db_name)
            if source_cfg is None:
                results.append(f"{source_name}: not configured or disabled")
                continue

            # Inject doc_ids if provided (for sources that support it)
            if doc_ids:
                source_cfg = {**source_cfg, "doc_ids": doc_ids}

            # Inject repos if provided (for github source)
            if repos:
                source_cfg = {**source_cfg, "repos": repos}

            # Inject branch if provided (for github source)
            if branch:
                source_cfg = {**source_cfg, "branch": branch}

            # Inject GitHub-specific options
            if include_issues:
                source_cfg = {**source_cfg, "include_issues": True}
            if include_prs:
                source_cfg = {**source_cfg, "include_prs": True}
            if include_wiki:
                source_cfg = {**source_cfg, "include_wiki": True}
            if include_source:
                source_cfg = {**source_cfg, "include_source": True}

            # Inject folder filter (for dropbox-paper)
            if folders:
                source_cfg = {**source_cfg, "folders": folders}

            # Inject channel filter (for slack)
            if channels:
                source_cfg = {**source_cfg, "channels": channels}

            try:
                source.configure(source_cfg)
            except Exception as e:
                results.append(f"{source_name}: config error - {e}")
                continue

            # Get sync state (unless full)
            state = None if full else _get_sync_state(conn, source_name, db_name)

            try:
                documents, new_state = source.fetch(state)
            except Exception as e:
                results.append(f"{source_name}: fetch error - {e}")
                continue

            if documents:
                ingester.ingest_documents(documents)
                results.append(f"{source_name}: synced {len(documents)} documents")
            else:
                results.append(f"{source_name}: no new documents")

            # Save state
            _save_sync_state(conn, source_name, db_name, new_state)

    return "\n".join(results)


def _run_rescan(
    db_url: str,
    dry_run: bool = False,
    delete_missing: bool = False,
) -> str:
    """Run rescan and return formatted result."""
    from .rescan import Rescanner

    rescanner = Rescanner(db_url, use_modal=True)
    result = rescanner.rescan(dry_run=dry_run, delete_missing=delete_missing, verbose=False)

    lines = []
    if dry_run:
        lines.append("(dry run - no changes made)")

    if result.updated:
        lines.append(f"Updated: {len(result.updated)} files")
        for path in result.updated[:5]:  # Show first 5
            lines.append(f"  - {path}")
        if len(result.updated) > 5:
            lines.append(f"  ... and {len(result.updated) - 5} more")

    if result.deleted:
        lines.append(f"Deleted: {len(result.deleted)} files")

    if result.missing:
        lines.append(f"Missing (not deleted): {len(result.missing)} files")
        for path in result.missing[:5]:
            lines.append(f"  - {path}")
        if len(result.missing) > 5:
            lines.append(f"  ... and {len(result.missing) - 5} more")

    lines.append(f"Unchanged: {result.unchanged} files")

    if result.errors:
        lines.append(f"Errors: {len(result.errors)}")
        for path, error in result.errors[:3]:
            lines.append(f"  - {path}: {error}")
        if len(result.errors) > 3:
            lines.append(f"  ... and {len(result.errors) - 3} more")

    return "\n".join(lines) if lines else "No indexed files found."


def _run_ingest(db_url: str, documents_data: list[dict]) -> str:
    """Ingest serialized documents and return formatted result."""
    import io
    from contextlib import redirect_stdout

    from .ingest import Document, Ingester

    documents = [Document.from_dict(d) for d in documents_data]
    ingester = Ingester(db_url, use_modal=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        ingester.ingest_documents(documents)

    output = buf.getvalue().strip()
    summary = f"Processed {len(documents)} document(s)."
    if output:
        return f"{summary}\n{output}"
    return summary


def _list_sync_sources(db_url: str, db_name: str) -> str:
    """List available sync sources with status and last sync time."""
    import psycopg
    from psycopg.rows import dict_row

    from .plugins.registry import PluginRegistry

    installed = PluginRegistry.list_sources()
    enabled = set(config.list_enabled_sources(db_name))

    if not installed:
        return "No API sync sources installed."

    # Get last sync times from database
    last_syncs = {}
    try:
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            results = conn.execute(
                """SELECT source_name, last_sync FROM sync_state WHERE database_name = %s""",
                (db_name,),
            ).fetchall()
            last_syncs = {r["source_name"]: r["last_sync"] for r in results}
    except Exception:
        pass  # Database may not be accessible

    lines = ["## API Sync Sources\n"]

    for name in sorted(installed):
        source = PluginRegistry.get_source(name)
        status = "enabled" if name in enabled else "disabled"
        source_type = source.source_type if source else "unknown"

        last_sync = last_syncs.get(name)
        if last_sync:
            last_sync_str = format_relative_time(last_sync.isoformat())
        else:
            last_sync_str = "never"

        lines.append(f"- **{name}** ({status}) - {source_type}")
        lines.append(f"  Last sync: {last_sync_str}")

    return "\n".join(lines)


def _synthesize_knowledge(
    db_url: str,
    project: str | None = None,
    sample_size: int = 25,
    max_proposals: int = 10,
) -> str:
    """Run synthesis to propose knowledge documents."""
    from .llm import get_llm
    from .llm.synthesize import synthesize

    if get_llm() is None:
        return (
            "Error: No LLM provider configured. "
            "Synthesis requires an LLM. Set ANTHROPIC_API_KEY or configure llm.provider in config."
        )

    try:
        result = synthesize(
            db_url=db_url,
            project=project,
            sample_size=sample_size,
            max_proposals=max_proposals,
            origin="mcp",
        )

        if not result.proposals:
            return "No proposals generated. The LLM response could not be parsed."

        lines = [f"Created {result.proposals_created} synthesis proposals:\n"]
        for p in result.proposals:
            lines.append(f"- **{p['title']}**")
            lines.append(f"  ID: `{p['id']}`")
        lines.append(
            "\nUse `list_pending_synthesis` to review, "
            "`approve_synthesis` or `reject_synthesis` to process."
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Error during synthesis: {e}"


def _list_pending_synthesis(
    db_url: str,
    project: str | None = None,
    limit: int = 50,
) -> str:
    """List pending synthesis proposals."""
    from .llm.synthesize import list_pending_synthesis

    proposals = list_pending_synthesis(db_url, project=project, limit=limit)

    if not proposals:
        return "No pending synthesis proposals."

    lines = ["## Pending Synthesis Proposals\n"]
    for p in proposals:
        lines.append(f"- **{p['title']}**")
        lines.append(f"  ID: `{p['id']}`")
        excerpt = p.get("excerpt", "")
        if excerpt:
            lines.append(f"  {excerpt}...")
        lines.append(f"  Model: {p['model']} | Origin: {p['origin']}")
        lines.append("")

    lines.append(f"{len(proposals)} pending proposals.")
    lines.append("Use `approve_synthesis` or `reject_synthesis` with the proposal ID.")
    return "\n".join(lines)


def _approve_synthesis(
    db_url: str,
    pending_id: str,
    title: str | None = None,
    content: str | None = None,
) -> str:
    """Approve a pending synthesis proposal."""
    from .llm.synthesize import approve_synthesis

    source_path = approve_synthesis(db_url, pending_id, title=title, content=content)
    if source_path:
        return f"Synthesis approved and created: `{source_path}`"
    return "Failed to approve. ID may be invalid or already processed."


def _reject_synthesis(db_url: str, pending_id: str) -> str:
    """Reject a pending synthesis proposal."""
    from .llm.synthesize import reject_synthesis

    if reject_synthesis(db_url, pending_id):
        return "Synthesis proposal rejected."
    return "Failed to reject. ID may be invalid or already processed."


def _edit_pending_synthesis(
    db_url: str,
    pending_id: str,
    title: str | None = None,
    content: str | None = None,
) -> str:
    """Edit a pending synthesis proposal."""
    from .llm.synthesize import update_pending_synthesis

    if update_pending_synthesis(db_url, pending_id, title=title, content=content):
        return "Synthesis proposal updated."
    return "Failed to update. ID may be invalid or already processed."


def _analyze_knowledge_base(
    db_url: str,
    project: str | None = None,
    sample_size: int = 15,
    auto_update: bool = True,
) -> str:
    """Analyze the knowledge base and return formatted result."""
    from .llm import get_llm
    from .llm.analyze import analyze_database, format_analysis_result

    # Check LLM is configured
    if get_llm() is None:
        return (
            "Error: No LLM provider configured. "
            "Analysis requires an LLM. Set ANTHROPIC_API_KEY or configure llm.provider in config."
        )

    try:
        result = analyze_database(
            db_url=db_url,
            project=project,
            sample_size=sample_size,
            auto_update=auto_update,
        )
        return format_analysis_result(result)
    except Exception as e:
        return f"Error analyzing knowledge base: {e}"


def _get_synthesis_samples(
    db_url: str,
    project: str | None = None,
    sample_size: int = 20,
    strategy: str = "diverse",
    excerpt_length: int = 1500,
) -> str:
    """Get document samples formatted for LLM-driven synthesis."""
    from .llm.analyze import get_content_stats, get_document_samples

    # Cap parameters
    sample_size = min(sample_size, 50)
    excerpt_length = min(excerpt_length, 3000)

    try:
        stats = get_content_stats(db_url, project)
        samples = get_document_samples(
            db_url,
            project=project,
            sample_size=sample_size,
            strategy=strategy,
            excerpt_length=excerpt_length,
        )
    except Exception as e:
        return f"Error fetching samples: {e}"

    # Format stats section
    lines = ["## Knowledge Base Statistics\n"]
    lines.append(f"- Total documents: {stats['total_documents']}")
    lines.append(f"- Total tokens: ~{stats['total_tokens']:,}")
    if stats["source_types"]:
        types_str = ", ".join(
            f"{t}: {c}" for t, c in sorted(stats["source_types"].items(), key=lambda x: -x[1])
        )
        lines.append(f"- Source types: {types_str}")
    if stats["projects"]:
        lines.append(f"- Projects: {', '.join(stats['projects'])}")
    if stats["date_range"]["earliest"]:
        lines.append(
            f"- Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}"
        )
    if project:
        lines.append(f"- Scoped to project: {project}")

    # Format samples section
    lines.append(f"\n## Document Samples ({len(samples)} documents)\n")
    for i, s in enumerate(samples, 1):
        lines.append(f"### {i}. {s['title']} ({s['source_type']})")
        excerpt = s["excerpt"] or ""
        lines.append(excerpt)
        lines.append("")

    return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _save_snapshot(name: str | None = None, db_name: str | None = None) -> str:
    """Create a database snapshot."""
    import subprocess

    from .config import get_snapshots_dir

    db_cfg = config.get_database(db_name)

    if not db_cfg.managed:
        return (
            f"Error: database '{db_cfg.name}' is not managed by okb "
            "(external databases not supported)"
        )

    # Check container is running
    try:
        result = subprocess.run(
            [
                "docker", "container", "inspect", "-f", "{{.State.Status}}",
                config.docker_container_name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or result.stdout.strip() != "running":
            return "Error: database container is not running"
    except Exception as e:
        return f"Error checking container status: {e}"

    # Generate name if not provided
    if not name:
        name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    snapshots_dir = get_snapshots_dir(db_cfg.database_name)
    snapshot_path = snapshots_dir / f"{name}.dump"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    if snapshot_path.exists():
        return f"Error: snapshot '{name}' already exists"

    # Run pg_dump
    try:
        result = subprocess.run(
            [
                "docker", "exec", config.docker_container_name,
                "pg_dump", "-U", "knowledge", "-Fc", db_cfg.database_name,
            ],
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            return f"Error: pg_dump failed: {result.stderr.decode()}"

        snapshot_path.write_bytes(result.stdout)
        size = _format_size(snapshot_path.stat().st_size)
        return f"Created snapshot '{name}' ({size})\nPath: {snapshot_path}"
    except subprocess.TimeoutExpired:
        return "Error: pg_dump timed out (>10 minutes)"
    except Exception as e:
        return f"Error creating snapshot: {e}"


def _list_snapshots(db_name: str | None = None) -> str:
    """List available database snapshots."""
    from .config import get_snapshots_dir

    db_cfg = config.get_database(db_name)
    snapshots_dir = get_snapshots_dir(db_cfg.database_name)

    if not snapshots_dir.exists():
        return f"No snapshots for database '{db_cfg.database_name}'"

    snapshots = sorted(snapshots_dir.glob("*.dump"))
    if not snapshots:
        return f"No snapshots for database '{db_cfg.database_name}'"

    lines = [f"## Snapshots for '{db_cfg.database_name}'\n"]
    lines.append("| Name | Size | Created |")
    lines.append("|------|------|---------|")
    for snap in snapshots:
        stat = snap.stat()
        size = _format_size(stat.st_size)
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| {snap.stem} | {size} | {mtime} |")

    return "\n".join(lines)


def _restore_snapshot(name: str, confirm: bool, db_name: str | None = None) -> str:
    """Restore database from a snapshot."""
    import subprocess

    from .config import get_snapshots_dir

    if not confirm:
        return (
            "Error: restore requires explicit confirmation. "
            "Set confirm=true to proceed. WARNING: This will replace ALL data in the database."
        )

    db_cfg = config.get_database(db_name)

    if not db_cfg.managed:
        return (
            f"Error: database '{db_cfg.name}' is not managed by okb "
            "(external databases not supported)"
        )

    # Check container is running
    try:
        result = subprocess.run(
            [
                "docker", "container", "inspect", "-f", "{{.State.Status}}",
                config.docker_container_name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or result.stdout.strip() != "running":
            return "Error: database container is not running"
    except Exception as e:
        return f"Error checking container status: {e}"

    snapshot_path = get_snapshots_dir(db_cfg.database_name) / f"{name}.dump"
    if not snapshot_path.exists():
        return f"Error: snapshot '{name}' not found"

    # Create pre-restore backup (always for MCP - safety first for LLM agents)
    backup_name = f"pre-restore-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    backup_result = _save_snapshot(backup_name, db_name=db_name)
    backup_warning = ""
    if backup_result.startswith("Error"):
        backup_warning = f" Warning: pre-restore backup failed ({backup_result})"
        backup_name = None

    # Run pg_restore
    try:
        snapshot_data = snapshot_path.read_bytes()
        result = subprocess.run(
            [
                "docker", "exec", "-i", config.docker_container_name,
                "pg_restore", "-U", "knowledge", "-d", db_cfg.database_name,
                "--clean", "--if-exists",
            ],
            input=snapshot_data,
            capture_output=True,
            timeout=600,
        )
        # pg_restore may return warnings even on success
        if result.returncode != 0 and b"error" in result.stderr.lower():
            return f"Error: pg_restore failed: {result.stderr.decode()}"

        msg = f"Restored database '{db_cfg.database_name}' from snapshot '{name}'"
        if backup_name:
            msg += f". Pre-restore backup saved as '{backup_name}'"
        if backup_warning:
            msg += f".{backup_warning}"
        return msg
    except subprocess.TimeoutExpired:
        return "Error: pg_restore timed out (>10 minutes)"
    except Exception as e:
        return f"Error restoring snapshot: {e}"


def build_server_instructions(db_config) -> str:
    """Build server instructions from database config and LLM metadata."""
    from . import __version__

    parts = [f"OKB v{__version__}"]
    if db_config.description:
        parts.append(db_config.description)
    if db_config.topics:
        parts.append(f"Topics: {', '.join(db_config.topics)}")
    return " ".join(parts)


# Initialize server and knowledge base
server = Server("knowledge-base", version=_okb_version)
kb: KnowledgeBase | None = None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Define available tools for Claude Code."""
    return [
        Tool(
            name="search_knowledge",
            description=(
                "Search the personal knowledge base for relevant information using semantic search. "
                "Use this for finding notes, code snippets, documentation, or any previously indexed content. "
                "Supports natural language queries - describe what you're looking for."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what you're looking for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5, max: 20)",
                        "default": 5,
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["markdown", "code"],
                        "description": "Filter by source type (optional)",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name (optional)",
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Filter to documents modified since. Accepts relative "
                            "('7d', '30d', '6mo', '1y'), natural language "
                            "('last week', '3 months ago'), or ISO date."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="keyword_search",
            description=(
                "Search by exact keywords using full-text search. "
                "Better for code symbols, function names, class names, or specific terms "
                "that semantic search might miss."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for (e.g., 'select_related prefetch')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                        "default": 5,
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["markdown", "code"],
                        "description": "Filter by source type (optional)",
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Filter to documents modified since. Accepts relative "
                            "('7d', '30d', '6mo', '1y'), natural language "
                            "('last week', '3 months ago'), or ISO date."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="hybrid_search",
            description=(
                "Combined semantic and keyword search using Reciprocal Rank Fusion. "
                "Use this when you want the best of both approaches - semantic understanding "
                "plus exact matching."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                        "default": 5,
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["markdown", "code"],
                        "description": "Filter by source type (optional)",
                    },
                    "since": {
                        "type": "string",
                        "description": (
                            "Filter to documents modified since. Accepts relative "
                            "('7d', '30d', '6mo', '1y'), natural language "
                            "('last week', '3 months ago'), or ISO date."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_document",
            description=(
                "Retrieve the full content of a specific document by its source path. "
                "Use after finding relevant chunks to get complete context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Absolute path to the document (from search results)",
                    },
                },
                "required": ["source_path"],
            },
        ),
        Tool(
            name="list_sources",
            description="List all indexed source types with document and chunk counts.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_projects",
            description="List all known project names in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_project_stats",
            description=(
                "Get projects with document counts. Use this to identify projects that should "
                "be consolidated (similar names, typos, etc.)."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_documents_by_project",
            description="List all documents belonging to a specific project.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name to list documents for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum documents to return (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["project"],
            },
        ),
        Tool(
            name="rename_project",
            description=(
                "Rename a project, updating all documents. Use for consolidating similar "
                "project names (e.g., 'my-app' and 'MyApp' -> 'my-app'). Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "old_name": {
                        "type": "string",
                        "description": "Current project name to rename",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New project name",
                    },
                },
                "required": ["old_name", "new_name"],
            },
        ),
        Tool(
            name="set_document_project",
            description=(
                "Set or clear the project for a single document. Use to fix incorrectly "
                "categorized documents. Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Path of the document to update",
                    },
                    "project": {
                        "type": "string",
                        "description": "New project name (omit or null to clear project)",
                    },
                },
                "required": ["source_path"],
            },
        ),
        Tool(
            name="recent_documents",
            description="Get recently indexed or updated documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of documents (default: 10)",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="save_knowledge",
            description=(
                "Save a piece of knowledge to the knowledge base for future reference. "
                "Use this to remember solutions, patterns, debugging tips, architectural decisions, "
                "or any useful information discovered during this conversation. "
                "The knowledge will be searchable in future sessions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short descriptive title for this knowledge",
                    },
                    "content": {
                        "type": "string",
                        "description": "The knowledge content to save",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categorization tags (e.g., ['python', 'debugging', 'django'])",
                    },
                    "project": {
                        "type": "string",
                        "description": "Associated project name (optional)",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["claude-note", "synthesis"],
                        "description": (
                            "Document type: 'claude-note' (default) for general knowledge, "
                            "'synthesis' for synthesis documents (uses okb://synthesis/ path)"
                        ),
                        "default": "claude-note",
                    },
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="delete_knowledge",
            description=(
                "Delete a document from the knowledge base by its source path. "
                "Works for any document type. Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "The source path of the knowledge entry to delete",
                    },
                },
                "required": ["source_path"],
            },
        ),
        Tool(
            name="update_knowledge",
            description=(
                "Update an existing knowledge document in-place. Preserves the source_path "
                "and created_at timestamp. Only provide fields you want to change; omitted "
                "fields keep their current values. Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "The source path of the document to update",
                    },
                    "title": {
                        "type": "string",
                        "description": "New title (omit to keep current)",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content, full replacement (omit to keep current)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags, replaces existing (omit to keep current)",
                    },
                    "project": {
                        "type": "string",
                        "description": "New project name (omit to keep current)",
                    },
                },
                "required": ["source_path"],
            },
        ),
        Tool(
            name="get_actionable_items",
            description=(
                "Query actionable items like tasks, calendar events, and emails "
                "with structured filters. Use this for daily briefs, finding tasks due soon, "
                "or checking today's schedule. Filters by status, due date, event date, priority."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "item_type": {
                        "type": "string",
                        "description": (
                            "Filter by source type (e.g., 'todoist-task', 'gcal-event')"
                        ),
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status ('pending', 'completed', 'cancelled')",
                    },
                    "due_date": {
                        "type": "string",
                        "description": (
                            "Filter tasks by due date: 'today', 'tomorrow', 'yesterday', "
                            "'this week', 'next week', 'last week', 'this month', "
                            "'next month', natural language, or YYYY-MM-DD"
                        ),
                    },
                    "event_date": {
                        "type": "string",
                        "description": (
                            "Filter events by date: 'today', 'tomorrow', 'yesterday', "
                            "'this week', 'next week', 'last week', 'this month', "
                            "'next month', natural language, or YYYY-MM-DD"
                        ),
                    },
                    "min_priority": {
                        "type": "integer",
                        "description": (
                            "Filter by priority (1=highest). Returns items <= this value."
                        ),
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 20)",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="get_database_info",
            description=(
                "Get information about this knowledge base including its description, topics, "
                "and content statistics. Call this at the start of a session to understand what's "
                "available. If the description/topics are empty or seem outdated, you SHOULD "
                "explore the database (list_sources, recent_documents, sample searches) and call "
                "set_database_description to document it for future sessions."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="set_database_description",
            description=(
                "Update the knowledge base description and topics based on your analysis of "
                "its contents. Use this after exploring the database to help future sessions "
                "understand what kind of information is stored here. Describe the content and "
                "purpose, not just stats. Good: 'Django backend for education platform with "
                "student enrollment and grading'. Bad: '2500 code files, 63 markdown docs'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "A concise description of what this knowledge base contains "
                            "(1-3 sentences, e.g., 'Personal notes on farming, including crop "
                            "planning, livestock management, and equipment maintenance')"
                        ),
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of topic keywords that characterize the content "
                            "(e.g., ['farming', 'crops', 'livestock', 'equipment'])"
                        ),
                    },
                },
            },
        ),
        Tool(
            name="add_todo",
            description=(
                "Create a TODO item in the knowledge base. Use this to capture tasks, "
                "action items, or reminders that come up during conversation. "
                "The TODO will be queryable via get_actionable_items."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "TODO item title",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional description or notes",
                    },
                    "due_date": {
                        "type": "string",
                        "description": (
                            "Due date: 'today', 'tomorrow', natural "
                            "language, or YYYY-MM-DD"
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority: 'A'/'B'/'C' or 1-5 (1=highest)",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categorization tags",
                    },
                },
                "required": ["title"],
            },
        ),
        Tool(
            name="trigger_sync",
            description=(
                "Trigger sync of API sources (Todoist, GitHub, Dropbox Paper, etc.). "
                "Fetches new/updated content from external services. Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of source names to sync (e.g., ['todoist', 'github']). "
                            "If empty and 'all' is false, returns list of available sources."
                        ),
                    },
                    "all": {
                        "type": "boolean",
                        "default": False,
                        "description": "Sync all enabled sources",
                    },
                    "full": {
                        "type": "boolean",
                        "default": False,
                        "description": "Ignore incremental state and do full resync",
                    },
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Specific document IDs to sync (for dropbox-paper). "
                            "If provided, only these documents are synced."
                        ),
                    },
                    "repos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "GitHub repos to sync (owner/repo format, can specify multiple). "
                            "Required for github source if not pre-configured in config."
                        ),
                    },
                    "branch": {
                        "type": "string",
                        "description": (
                            "Git branch to sync (default: repo default branch). "
                            "Applies to github source."
                        ),
                    },
                    "include_issues": {
                        "type": "boolean",
                        "description": "Include GitHub issues in sync.",
                    },
                    "include_prs": {
                        "type": "boolean",
                        "description": "Include GitHub pull requests in sync.",
                    },
                    "include_wiki": {
                        "type": "boolean",
                        "description": "Include GitHub wiki pages in sync.",
                    },
                    "include_source": {
                        "type": "boolean",
                        "description": (
                            "Sync all source files, not just README + docs/. "
                            "Applies to github source."
                        ),
                    },
                    "folders": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Filter to specific folder paths (for dropbox-paper)."
                        ),
                    },
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Slack channel IDs to sync.",
                    },
                },
            },
        ),
        Tool(
            name="trigger_rescan",
            description=(
                "Check indexed files for changes and re-ingest stale ones. "
                "Compares stored modification times with current filesystem. "
                "Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Only report what would change, don't actually re-ingest",
                    },
                    "delete_missing": {
                        "type": "boolean",
                        "default": False,
                        "description": "Remove documents for files that no longer exist",
                    },
                },
            },
        ),
        Tool(
            name="ingest_documents",
            description=(
                "Ingest pre-parsed documents into the knowledge base. "
                "Documents should be serialized with source_path, source_type, "
                "title, content, and optional metadata/sections. The server handles "
                "chunking, embedding, and storage. Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_path": {"type": "string"},
                                "source_type": {"type": "string"},
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "metadata": {"type": "object"},
                                "sections": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "due_date": {"type": "string"},
                                "status": {"type": "string"},
                                "priority": {"type": "integer"},
                            },
                            "required": [
                                "source_path", "source_type", "title", "content",
                            ],
                        },
                        "description": "Array of serialized Document objects",
                    },
                },
                "required": ["documents"],
            },
        ),
        Tool(
            name="list_sync_sources",
            description=(
                "List available API sync sources (Todoist, GitHub, Dropbox Paper, etc.) "
                "with their enabled/disabled status and last sync time. "
                "Use this to see what external data sources can be synced."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="synthesize_knowledge",
            description=(
                "Analyze the knowledge base and propose synthetic knowledge documents. "
                "Generates topic summaries, entity profiles, relationship maps, and insights. "
                "Proposals go to pending review. Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Scope synthesis to a specific project (optional)",
                    },
                    "sample_size": {
                        "type": "integer",
                        "default": 25,
                        "description": "Number of documents to sample (default: 25)",
                    },
                    "max_proposals": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum proposals to generate (default: 10)",
                    },
                },
            },
        ),
        Tool(
            name="list_pending_synthesis",
            description=(
                "List pending synthesis proposals awaiting review. "
                "Use approve_synthesis or reject_synthesis to process them."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Filter by project (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum results",
                    },
                },
            },
        ),
        Tool(
            name="approve_synthesis",
            description=(
                "Approve a pending synthesis proposal, creating it as a searchable document. "
                "Optionally provide edited title/content. Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pending_id": {
                        "type": "string",
                        "description": "ID of the pending synthesis to approve",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title override",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional content override",
                    },
                },
                "required": ["pending_id"],
            },
        ),
        Tool(
            name="reject_synthesis",
            description=(
                "Reject a pending synthesis proposal. "
                "Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pending_id": {
                        "type": "string",
                        "description": "ID of the pending synthesis to reject",
                    },
                },
                "required": ["pending_id"],
            },
        ),
        Tool(
            name="edit_pending_synthesis",
            description=(
                "Edit a pending synthesis proposal before approving or rejecting. "
                "Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pending_id": {
                        "type": "string",
                        "description": "ID of the pending synthesis to edit",
                    },
                    "title": {
                        "type": "string",
                        "description": "New title",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content",
                    },
                },
                "required": ["pending_id"],
            },
        ),
        Tool(
            name="analyze_knowledge_base",
            description=(
                "Analyze the knowledge base to generate or update its description and topics. "
                "Uses document samples to understand themes and content. "
                "Results are stored in database_metadata for future sessions. "
                "Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Analyze only a specific project (optional)",
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "Number of documents to sample (default: 15)",
                        "default": 15,
                    },
                    "auto_update": {
                        "type": "boolean",
                        "description": "Update database metadata with results (default: true)",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="get_synthesis_samples",
            description=(
                "Get document samples and statistics for synthesizing knowledge. "
                "Returns content excerpts suitable for the calling LLM to analyze and produce "
                "synthesis documents. Save results via save_knowledge with source_type='synthesis'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Scope to a specific project (optional)",
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "Number of documents to sample (default: 20, max: 50)",
                        "default": 20,
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["diverse", "recent", "random"],
                        "description": "Sampling strategy (default: diverse)",
                        "default": "diverse",
                    },
                    "excerpt_length": {
                        "type": "integer",
                        "description": "Characters per excerpt (default: 1500, max: 3000)",
                        "default": 1500,
                    },
                },
            },
        ),
        Tool(
            name="save_snapshot",
            description=(
                "Create a database snapshot for backup or migration. "
                "Only works for managed (Docker) databases. Requires write permission."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Snapshot name (default: timestamp like 20250204T143022)",
                    },
                },
            },
        ),
        Tool(
            name="list_snapshots",
            description="List available database snapshots with metadata (size, date).",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="restore_snapshot",
            description=(
                "Restore database from a snapshot. WARNING: This replaces ALL data. "
                "Only works for managed databases. Requires write permission and confirm=true."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the snapshot to restore",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to proceed with restore (safety check)",
                    },
                },
                "required": ["name", "confirm"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool invocations from Claude Code."""
    from .tools import execute_tool

    db_config = config.get_database()
    return await execute_tool(kb, name, arguments, db_name=db_config.name, db_config=db_config)


async def main(db_url: str | None = None, db_name: str | None = None):
    """Run the MCP server."""
    global kb

    # Get database config
    db_config = config.get_database(db_name)

    # Initialize knowledge base with provided URL or from config
    if db_url is None:
        db_url = db_config.url
    kb = KnowledgeBase(db_url)

    # Set server instructions from config
    server.instructions = build_server_instructions(db_config)

    # Pre-warm embedding model
    print("Warming up embedding model...", file=sys.stderr)
    warmup()
    print("Ready.", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
