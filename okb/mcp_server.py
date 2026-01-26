"""
MCP Server for Knowledge Base.

Exposes semantic search to Claude Code via the Model Context Protocol.

Usage:
    python mcp_server.py

Configure in Claude Code (~/.claude.json or similar):
    {
      "mcpServers": {
        "knowledge-base": {
          "command": "python",
          "args": ["/path/to/mcp_server.py"]
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

import psycopg
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .config import config
from .local_embedder import embed_document, embed_query, warmup


def get_document_date(metadata: dict) -> str | None:
    """Get best available date: document_date > file_modified_at."""
    return metadata.get("document_date") or metadata.get("file_modified_at")


def format_relative_time(iso_timestamp: str) -> str:
    """Format ISO timestamp as relative time (e.g., '3d ago')."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        # Handle naive datetimes (date-only strings like '2020-11-10')
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - dt
        if delta.days < 0:
            return "future"
        if delta.days > 365:
            return f"{delta.days // 365}y ago"
        if delta.days > 30:
            return f"{delta.days // 30}mo ago"
        if delta.days > 0:
            return f"{delta.days}d ago"
        if delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        if delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        return "just now"
    except (ValueError, TypeError):
        return ""


def parse_since_filter(since: str) -> datetime | None:
    """Parse since filter like '7d', '30d', '6mo' or ISO date."""
    from datetime import timedelta

    now = datetime.now(UTC)
    match = re.match(r"^(\d+)(d|mo|y)$", since.lower())
    if match:
        value, unit = int(match.group(1)), match.group(2)
        days = value * {"d": 1, "mo": 30, "y": 365}[unit]
        return now - timedelta(days=days)
    try:
        return datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_date_range(date_str: str) -> tuple[datetime, datetime] | None:
    """Parse date range like 'today', 'tomorrow', 'this_week', '2024-01-15', or ISO date."""
    from datetime import timedelta

    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    if date_str.lower() == "today":
        return (today_start, today_end)
    elif date_str.lower() == "tomorrow":
        return (today_end, today_end + timedelta(days=1))
    elif date_str.lower() == "this_week":
        # Monday to Sunday
        days_since_monday = now.weekday()
        week_start = today_start - timedelta(days=days_since_monday)
        return (week_start, week_start + timedelta(days=7))
    elif date_str.lower() == "next_week":
        days_since_monday = now.weekday()
        next_week_start = today_start + timedelta(days=7 - days_since_monday)
        return (next_week_start, next_week_start + timedelta(days=7))
    elif re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        # Single date: return that day
        try:
            dt = datetime.fromisoformat(date_str).replace(tzinfo=UTC)
            return (dt, dt + timedelta(days=1))
        except ValueError:
            return None
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
    ) -> dict:
        """
        Save a piece of knowledge directly from Claude.

        Creates a virtual document (not file-backed) with embedding.
        Returns the saved document info.
        """
        conn = self.get_connection()

        # Generate unique source path for Claude-generated content
        knowledge_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        source_path = f"claude://knowledge/{timestamp}-{knowledge_id}"

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
                "claude-note",
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

    def delete_knowledge(self, source_path: str) -> bool:
        """Delete a Claude-saved knowledge entry by source path."""
        if not source_path.startswith("claude://"):
            return False

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
) -> str:
    """Run sync for specified sources and return formatted result."""
    from psycopg.rows import dict_row

    from .ingest import Ingester
    from .plugins.registry import PluginRegistry

    # Determine which sources to sync
    if sync_all:
        source_names = config.list_enabled_sources()
    elif sources:
        source_names = list(sources)
    else:
        # Return list of available sources
        installed = PluginRegistry.list_sources()
        configured = config.list_enabled_sources()
        lines = ["Available API sources:"]
        for name in installed:
            status = "enabled" if name in configured else "disabled"
            lines.append(f"  - {name} ({status})")
        if not installed:
            lines.append("  (none installed)")
        return "\n".join(lines)

    if not source_names:
        return "No sources to sync."

    # Get database name from URL for sync state
    db_name = config.get_database().name

    results = []
    ingester = Ingester(db_url, use_modal=True)

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        for source_name in source_names:
            # Get the plugin
            source = PluginRegistry.get_source(source_name)
            if source is None:
                results.append(f"{source_name}: not found")
                continue

            # Get and resolve config
            source_cfg = config.get_source_config(source_name)
            if source_cfg is None:
                results.append(f"{source_name}: not configured or disabled")
                continue

            # Inject doc_ids if provided (for sources that support it)
            if doc_ids:
                source_cfg = {**source_cfg, "doc_ids": doc_ids}

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


def build_server_instructions(db_config) -> str | None:
    """Build server instructions from database config and LLM metadata."""
    parts = []
    if db_config.description:
        parts.append(db_config.description)
    if db_config.topics:
        parts.append(f"Topics: {', '.join(db_config.topics)}")
    return " ".join(parts) if parts else None


# Initialize server and knowledge base
server = Server("knowledge-base")
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
                        "description": "Filter to documents modified since (ISO date or relative: '7d', '30d', '6mo')",
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
                        "description": "Filter to documents modified since (ISO date or relative: '7d', '30d', '6mo')",
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
                        "description": "Filter to documents modified since (ISO date or relative: '7d', '30d', '6mo')",
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
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="delete_knowledge",
            description=(
                "Delete a previously saved knowledge entry by its source path. "
                "Only works for Claude-saved entries (claude:// paths)."
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
                            "Filter tasks by due date: 'today', 'tomorrow', 'this_week', "
                            "'next_week', or 'YYYY-MM-DD'"
                        ),
                    },
                    "event_date": {
                        "type": "string",
                        "description": (
                            "Filter events by date: 'today', 'tomorrow', 'this_week', "
                            "'next_week', or 'YYYY-MM-DD'"
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
                        "description": ("Due date: ISO date (YYYY-MM-DD), 'today', or 'tomorrow'"),
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
    ]


def format_search_results(results: list[dict], show_similarity: bool = True) -> str:
    """Format search results for display."""
    if not results:
        return "No relevant results found."

    output = []
    for r in results:
        header = f"## {r['title']} ({r['source_type']})"
        source = f"**Source:** `{r['source_path']}`"

        # Add document date if available
        date_line = ""
        if doc_meta := r.get("doc_metadata"):
            if doc_date := get_document_date(doc_meta):
                date_line = f"\n**Modified:** {format_relative_time(doc_date)}"

        if show_similarity and "similarity" in r:
            score = f"**Relevance:** {r['similarity']:.1%}"
            output.append(f"{header}\n{source}\n{score}{date_line}\n\n{r['content']}\n\n---")
        elif "rank" in r:
            output.append(f"{header}\n{source}{date_line}\n\n{r['content']}\n\n---")
        else:
            output.append(f"{header}\n{source}{date_line}\n\n{r['content']}\n\n---")

    return "\n\n".join(output)


def format_actionable_items(items: list[dict]) -> str:
    """Format actionable items (tasks, events, emails) for display."""
    if not items:
        return "No actionable items found matching the criteria."

    output = ["## Actionable Items\n"]

    for item in items:
        title = item.get("title") or "Untitled"
        source_type = item.get("source_type", "unknown")
        status = item.get("status")
        priority = item.get("priority")

        # Build header with status and priority indicators
        status_icon = {"pending": "[ ]", "completed": "[x]", "cancelled": "[-]"}.get(
            status or "", "[ ]"
        )
        priority_str = f" P{priority}" if priority else ""
        header = f"{status_icon} **{title}**{priority_str} ({source_type})"

        # Build date info
        date_parts = []
        if due := item.get("due_date"):
            date_parts.append(f"Due: {format_relative_time(str(due))}")
        if start := item.get("event_start"):
            if end := item.get("event_end"):
                date_parts.append(f"Event: {start} - {end}")
            else:
                date_parts.append(f"Event: {start}")
        date_line = " | ".join(date_parts) if date_parts else ""

        # Content preview (truncate if long)
        content = item.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."

        parts = [header]
        if date_line:
            parts.append(date_line)
        if content:
            parts.append(content)
        parts.append(f"`{item.get('source_path', '')}`")
        parts.append("---")

        output.append("\n".join(parts))

    return "\n\n".join(output)


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool invocations from Claude Code."""
    try:
        if name == "search_knowledge":
            results = kb.semantic_search(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type=arguments.get("source_type"),
                project=arguments.get("project"),
                since=arguments.get("since"),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=format_search_results(results))]
            )

        elif name == "keyword_search":
            results = kb.keyword_search(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type=arguments.get("source_type"),
                since=arguments.get("since"),
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=format_search_results(results, show_similarity=False)
                    )
                ]
            )

        elif name == "hybrid_search":
            results = kb.hybrid_search(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type=arguments.get("source_type"),
                since=arguments.get("since"),
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=format_search_results(results, show_similarity=False)
                    )
                ]
            )

        elif name == "get_document":
            doc = kb.get_document(arguments["source_path"])
            if not doc:
                return CallToolResult(
                    content=[TextContent(type="text", text="Document not found.")]
                )
            return CallToolResult(
                content=[TextContent(type="text", text=f"# {doc['title']}\n\n{doc['content']}")]
            )

        elif name == "list_sources":
            sources = kb.list_sources()
            if not sources:
                return CallToolResult(
                    content=[TextContent(type="text", text="No documents indexed yet.")]
                )
            output = ["## Indexed Sources\n"]
            for s in sources:
                tokens = s.get("total_tokens") or 0
                output.append(
                    f"- **{s['source_type']}**: {s['document_count']} documents, "
                    f"{s['chunk_count']} chunks (~{tokens:,} tokens)"
                )
            return CallToolResult(content=[TextContent(type="text", text="\n".join(output))])

        elif name == "list_projects":
            projects = kb.list_projects()
            if not projects:
                return CallToolResult(content=[TextContent(type="text", text="No projects found.")])
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text="## Projects\n\n" + "\n".join(f"- {p}" for p in projects)
                    )
                ]
            )

        elif name == "recent_documents":
            docs = kb.get_recent_documents(arguments.get("limit", 10))
            if not docs:
                return CallToolResult(
                    content=[TextContent(type="text", text="No documents indexed yet.")]
                )
            output = ["## Recent Documents\n"]
            for d in docs:
                project = d["metadata"].get("project", "")
                project_str = f" [{project}]" if project else ""
                date_str = ""
                if doc_date := get_document_date(d["metadata"]):
                    date_str = f" - {format_relative_time(doc_date)}"
                output.append(f"- **{d['title']}**{project_str} ({d['source_type']}){date_str}")
                output.append(f"  `{d['source_path']}`")
            return CallToolResult(content=[TextContent(type="text", text="\n".join(output))])

        elif name == "save_knowledge":
            result = kb.save_knowledge(
                title=arguments["title"],
                content=arguments["content"],
                tags=arguments.get("tags"),
                project=arguments.get("project"),
            )
            if result["status"] == "duplicate":
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                f"Duplicate content already exists:\n"
                                f"- Title: {result['existing_title']}\n"
                                f"- Path: `{result['existing_path']}`"
                            ),
                        )
                    ]
                )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"Knowledge saved successfully:\n"
                            f"- Title: {result['title']}\n"
                            f"- Path: `{result['source_path']}`\n"
                            f"- Tokens: ~{result['token_count']}"
                        ),
                    )
                ]
            )

        elif name == "delete_knowledge":
            deleted = kb.delete_knowledge(arguments["source_path"])
            if deleted:
                return CallToolResult(
                    content=[TextContent(type="text", text="Knowledge entry deleted.")]
                )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Could not delete. Entry not found or not a Claude-saved entry.",
                    )
                ]
            )

        elif name == "get_actionable_items":
            items = kb.get_actionable_items(
                item_type=arguments.get("item_type"),
                status=arguments.get("status"),
                due_date=arguments.get("due_date"),
                event_date=arguments.get("event_date"),
                min_priority=arguments.get("min_priority"),
                limit=arguments.get("limit", 20),
            )
            return CallToolResult(
                content=[TextContent(type="text", text=format_actionable_items(items))]
            )

        elif name == "get_database_info":
            # Get config-based info
            db_config = config.get_database()
            info_parts = ["## Knowledge Base Info\n"]

            # Config-defined description/topics
            if db_config.description:
                info_parts.append(f"**Description (config):** {db_config.description}")
            if db_config.topics:
                info_parts.append(f"**Topics (config):** {', '.join(db_config.topics)}")

            # LLM-enhanced metadata
            llm_desc = None
            llm_topics = None
            try:
                metadata = kb.get_database_metadata()
                llm_desc = metadata.get("llm_description", {}).get("value")
                llm_topics = metadata.get("llm_topics", {}).get("value")
                if llm_desc:
                    info_parts.append(f"**Description (LLM-enhanced):** {llm_desc}")
                if llm_topics:
                    info_parts.append(f"**Topics (LLM-enhanced):** {', '.join(llm_topics)}")
            except Exception:
                pass  # Table may not exist yet

            # Content stats
            sources = kb.list_sources()
            if sources:
                info_parts.append("\n### Content Statistics")
                for s in sources:
                    tokens = s.get("total_tokens") or 0
                    info_parts.append(
                        f"- **{s['source_type']}**: {s['document_count']} documents, "
                        f"{s['chunk_count']} chunks (~{tokens:,} tokens)"
                    )

            # Projects
            projects = kb.list_projects()
            if projects:
                info_parts.append(f"\n### Projects\n{', '.join(projects)}")

            # Hint if no description exists
            if not db_config.description and not llm_desc:
                info_parts.append(
                    "\n**Note:** No description set. Consider exploring with "
                    "recent_documents and search_knowledge, then calling "
                    "set_database_description to document what's here."
                )

            return CallToolResult(content=[TextContent(type="text", text="\n".join(info_parts))])

        elif name == "set_database_description":
            updated = []
            if "description" in arguments:
                kb.set_database_metadata("llm_description", arguments["description"])
                updated.append("description")
            if "topics" in arguments:
                kb.set_database_metadata("llm_topics", arguments["topics"])
                updated.append("topics")
            if updated:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Updated database metadata: {', '.join(updated)}",
                        )
                    ]
                )
            return CallToolResult(
                content=[TextContent(type="text", text="No fields provided to update.")]
            )

        elif name == "add_todo":
            result = kb.save_todo(
                title=arguments["title"],
                content=arguments.get("content"),
                due_date=arguments.get("due_date"),
                priority=arguments.get("priority"),
                project=arguments.get("project"),
                tags=arguments.get("tags"),
            )
            parts = [
                "TODO created:",
                f"- Title: {result['title']}",
                f"- Path: `{result['source_path']}`",
            ]
            if result.get("priority"):
                parts.append(f"- Priority: P{result['priority']}")
            if result.get("due_date"):
                parts.append(f"- Due: {result['due_date']}")
            return CallToolResult(content=[TextContent(type="text", text="\n".join(parts))])

        elif name == "trigger_sync":
            result = _run_sync(
                kb.db_url,
                sources=arguments.get("sources", []),
                sync_all=arguments.get("all", False),
                full=arguments.get("full", False),
                doc_ids=arguments.get("doc_ids"),
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "trigger_rescan":
            result = _run_rescan(
                kb.db_url,
                dry_run=arguments.get("dry_run", False),
                delete_missing=arguments.get("delete_missing", False),
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])

    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Error: {e!s}")])


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
