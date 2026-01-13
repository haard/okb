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
import sys
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
from .local_embedder import embed_query, warmup


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

        sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
        params.extend([embedding, min(limit, config.max_limit)])

        results = conn.execute(sql, params).fetchall()
        return [dict(r) for r in results]

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        source_type: str | None = None,
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
                ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', %s)) as rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
        """
        params: list[Any] = [query, query]

        if source_type:
            sql += " AND d.source_type = %s"
            params.append(source_type)

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
    ) -> list[dict]:
        """
        Hybrid search combining semantic and keyword results.

        Uses Reciprocal Rank Fusion (RRF) to merge results.
        """
        # Get both result sets
        semantic_results = self.semantic_search(query, limit=limit * 2, source_type=source_type)
        keyword_results = self.keyword_search(query, limit=limit * 2, source_type=source_type)

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
    ]


def format_search_results(results: list[dict], show_similarity: bool = True) -> str:
    """Format search results for display."""
    if not results:
        return "No relevant results found."

    output = []
    for r in results:
        header = f"## {r['title']} ({r['source_type']})"
        source = f"**Source:** `{r['source_path']}`"

        if show_similarity and "similarity" in r:
            score = f"**Relevance:** {r['similarity']:.1%}"
            output.append(f"{header}\n{source}\n{score}\n\n{r['content']}\n\n---")
        elif "rank" in r:
            output.append(f"{header}\n{source}\n\n{r['content']}\n\n---")
        else:
            output.append(f"{header}\n{source}\n\n{r['content']}\n\n---")

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
            )
            return CallToolResult(
                content=[TextContent(type="text", text=format_search_results(results))]
            )

        elif name == "keyword_search":
            results = kb.keyword_search(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type=arguments.get("source_type"),
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
                output.append(f"- **{d['title']}**{project_str} ({d['source_type']})")
                output.append(f"  `{d['source_path']}`")
            return CallToolResult(content=[TextContent(type="text", text="\n".join(output))])

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
