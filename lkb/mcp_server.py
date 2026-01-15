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
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from .config import config
from .local_embedder import embed_document, embed_query, warmup


def get_document_date(metadata: dict) -> str | None:
    """Get best available date: document_date > file_modified_at."""
    return metadata.get("document_date") or metadata.get("file_modified_at")


def format_relative_time(iso_timestamp: str) -> str:
    """Format ISO timestamp as relative time (e.g., '3d ago')."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
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
    import re
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    match = re.match(r"^(\d+)(d|mo|y)$", since.lower())
    if match:
        value, unit = int(match.group(1)), match.group(2)
        days = value * {"d": 1, "mo": 30, "y": 365}[unit]
        return now - timedelta(days=days)
    try:
        return datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError:
        return None


class KnowledgeBase:
    """Knowledge base with semantic and keyword search."""

    def __init__(self):
        self._conn = None

    def get_connection(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(config.db_url, row_factory=dict_row)
            register_vector(self._conn)
        return self._conn

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
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
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


# Initialize server and knowledge base
server = Server("knowledge-base")
kb = KnowledgeBase()


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

        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])

    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Error: {e!s}")])


async def main():
    """Run the MCP server."""
    # Pre-warm embedding model
    print("Warming up embedding model...", file=sys.stderr)
    warmup()
    print("Ready.", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
