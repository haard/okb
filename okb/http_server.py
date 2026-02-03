"""HTTP transport server for MCP with token authentication.

This module provides an HTTP server that serves the LKB MCP server with
token-based authentication. Tokens can be passed via Authorization header
or query parameter. A single HTTP server can serve multiple databases,
with the token determining which database to use.
"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import CallToolResult, TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from .config import config
from .local_embedder import warmup
from .mcp_server import (
    KnowledgeBase,
    format_actionable_items,
    format_search_results,
)
from .tokens import OKBTokenVerifier, TokenInfo

# Permission sets
READ_ONLY_TOOLS = frozenset(
    {
        "search_knowledge",
        "keyword_search",
        "hybrid_search",
        "get_document",
        "list_sources",
        "list_projects",
        "list_documents_by_project",
        "recent_documents",
        "get_actionable_items",
        "get_database_info",
        "list_sync_sources",
        "list_pending_entities",
        "list_pending_merges",
        "get_topic_clusters",
        "get_entity_relationships",
    }
)

WRITE_TOOLS = frozenset(
    {
        "save_knowledge",
        "delete_knowledge",
        "set_database_description",
        "add_todo",
        "trigger_sync",
        "trigger_rescan",
        "enrich_document",
        "approve_entity",
        "reject_entity",
        "analyze_knowledge_base",
        "find_entity_duplicates",
        "merge_entities",
        "approve_merge",
        "reject_merge",
        "run_consolidation",
    }
)


def extract_token(request: Request) -> str | None:
    """Extract token from Authorization header or query parameter."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    if "token" in request.query_params:
        return request.query_params["token"]
    return None


class HTTPMCPServer:
    """HTTP server for MCP with token authentication."""

    def __init__(self):
        self.knowledge_bases: dict[str, KnowledgeBase] = {}
        self.server = Server("knowledge-base")
        # Single shared transport instance for all connections
        self.transport = SseServerTransport("/messages/")
        # Map session_id (hex string) -> token_info
        self.session_tokens: dict[str, TokenInfo] = {}
        self._setup_handlers()

    def _get_db_url(self, db_name: str) -> str:
        """Get database URL by name."""
        return config.get_database(db_name).url

    def _setup_handlers(self):
        """Set up MCP server handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Define available tools for Claude Code."""
            # Import the tool definitions from mcp_server
            from .mcp_server import list_tools as get_tools

            return await get_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            """Handle tool invocations with permission checking."""
            # Get auth context from the current request
            # This is passed via the transport
            token_info: TokenInfo | None = getattr(self.server, "_current_token_info", None)

            if token_info is None:
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: No authentication context")]
                )

            # Check permissions
            if name in WRITE_TOOLS and token_info.permissions == "ro":
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Error: Permission denied. Tool '{name}' requires write access.",
                        )
                    ]
                )

            # Get or create knowledge base for this database
            if token_info.database not in self.knowledge_bases:
                db_url = self._get_db_url(token_info.database)
                self.knowledge_bases[token_info.database] = KnowledgeBase(db_url)

            kb = self.knowledge_bases[token_info.database]

            # Execute the tool
            return await self._execute_tool(kb, name, arguments)

    async def _execute_tool(
        self, kb: KnowledgeBase, name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        """Execute a tool on a specific knowledge base."""
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
                    return CallToolResult(
                        content=[TextContent(type="text", text="No projects found.")]
                    )
                project_list = "\n".join(f"- {p}" for p in projects)
                return CallToolResult(
                    content=[TextContent(type="text", text=f"## Projects\n\n{project_list}")]
                )

            elif name == "list_documents_by_project":
                project = arguments["project"]
                limit = arguments.get("limit", 100)
                docs = kb.list_documents_by_project(project, limit)
                if not docs:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text", text=f"No documents found for project '{project}'."
                            )
                        ]
                    )
                output = [f"## Documents in '{project}' ({len(docs)} documents)\n"]
                for d in docs:
                    output.append(f"- **{d['title'] or d['source_path']}** ({d['source_type']})")
                    output.append(f"  - `{d['source_path']}`")
                return CallToolResult(content=[TextContent(type="text", text="\n".join(output))])

            elif name == "recent_documents":
                from .mcp_server import format_relative_time, get_document_date

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
                        content=[TextContent(type="text", text="Document deleted.")]
                    )
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="Could not delete. Document not found.",
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
                # Get config-based info for the token's database
                token_info = getattr(self.server, "_current_token_info", None)
                db_config = config.get_database(token_info.database if token_info else None)
                info_parts = ["## Knowledge Base Info\n"]

                if db_config.description:
                    info_parts.append(f"**Description (config):** {db_config.description}")
                if db_config.topics:
                    info_parts.append(f"**Topics (config):** {', '.join(db_config.topics)}")

                # LLM-enhanced metadata
                try:
                    metadata = kb.get_database_metadata()
                    llm_desc = metadata.get("llm_description", {}).get("value")
                    llm_topics = metadata.get("llm_topics", {}).get("value")
                    if llm_desc:
                        info_parts.append(f"**Description (LLM-enhanced):** {llm_desc}")
                    if llm_topics:
                        info_parts.append(f"**Topics (LLM-enhanced):** {', '.join(llm_topics)}")
                except Exception:
                    pass

                sources = kb.list_sources()
                if sources:
                    info_parts.append("\n### Content Statistics")
                    for s in sources:
                        tokens = s.get("total_tokens") or 0
                        info_parts.append(
                            f"- **{s['source_type']}**: {s['document_count']} documents, "
                            f"{s['chunk_count']} chunks (~{tokens:,} tokens)"
                        )

                projects = kb.list_projects()
                if projects:
                    info_parts.append(f"\n### Projects\n{', '.join(projects)}")

                return CallToolResult(
                    content=[TextContent(type="text", text="\n".join(info_parts))]
                )

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
                from .mcp_server import _run_sync

                # Get the db_url from the knowledge base
                result = _run_sync(
                    kb.db_url,
                    sources=arguments.get("sources", []),
                    sync_all=arguments.get("all", False),
                    full=arguments.get("full", False),
                    doc_ids=arguments.get("doc_ids"),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "trigger_rescan":
                from .mcp_server import _run_rescan

                result = _run_rescan(
                    kb.db_url,
                    dry_run=arguments.get("dry_run", False),
                    delete_missing=arguments.get("delete_missing", False),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "list_sync_sources":
                from .mcp_server import _list_sync_sources

                token_info = getattr(self.server, "_current_token_info", None)
                db_name = token_info.database if token_info else config.get_database().name
                result = _list_sync_sources(kb.db_url, db_name)
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "enrich_document":
                from .mcp_server import _enrich_document

                result = _enrich_document(
                    kb.db_url,
                    source_path=arguments["source_path"],
                    extract_todos=arguments.get("extract_todos", True),
                    extract_entities=arguments.get("extract_entities", True),
                    auto_create_entities=arguments.get("auto_create_entities", False),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "list_pending_entities":
                from .mcp_server import _list_pending_entities

                result = _list_pending_entities(
                    kb.db_url,
                    entity_type=arguments.get("entity_type"),
                    limit=arguments.get("limit", 20),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "approve_entity":
                from .mcp_server import _approve_entity

                result = _approve_entity(kb.db_url, arguments["pending_id"])
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "reject_entity":
                from .mcp_server import _reject_entity

                result = _reject_entity(kb.db_url, arguments["pending_id"])
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "analyze_knowledge_base":
                from .mcp_server import _analyze_knowledge_base

                result = _analyze_knowledge_base(
                    kb.db_url,
                    project=arguments.get("project"),
                    sample_size=arguments.get("sample_size", 15),
                    auto_update=arguments.get("auto_update", True),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            # Entity consolidation tools
            elif name == "find_entity_duplicates":
                from .mcp_server import _find_entity_duplicates

                result = _find_entity_duplicates(
                    kb.db_url,
                    similarity_threshold=arguments.get("similarity_threshold", 0.85),
                    limit=arguments.get("limit", 50),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "merge_entities":
                from .mcp_server import _merge_entities

                result = _merge_entities(
                    kb.db_url,
                    canonical_path=arguments["canonical_path"],
                    duplicate_path=arguments["duplicate_path"],
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "list_pending_merges":
                from .mcp_server import _list_pending_merges

                result = _list_pending_merges(
                    kb.db_url,
                    limit=arguments.get("limit", 50),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "approve_merge":
                from .mcp_server import _approve_merge

                result = _approve_merge(kb.db_url, arguments["merge_id"])
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "reject_merge":
                from .mcp_server import _reject_merge

                result = _reject_merge(kb.db_url, arguments["merge_id"])
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "get_topic_clusters":
                from .mcp_server import _get_topic_clusters

                result = _get_topic_clusters(
                    kb.db_url,
                    limit=arguments.get("limit", 20),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "get_entity_relationships":
                from .mcp_server import _get_entity_relationships

                result = _get_entity_relationships(
                    kb.db_url,
                    entity_name=arguments.get("entity_name"),
                    relationship_type=arguments.get("relationship_type"),
                    limit=arguments.get("limit", 50),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "run_consolidation":
                from .mcp_server import _run_consolidation

                result = _run_consolidation(
                    kb.db_url,
                    detect_duplicates=arguments.get("detect_duplicates", True),
                    detect_cross_doc=arguments.get("detect_cross_doc", True),
                    build_clusters=arguments.get("build_clusters", True),
                    extract_relationships=arguments.get("extract_relationships", True),
                    dry_run=arguments.get("dry_run", False),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                )

        except Exception as e:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: {e!s}")])

    def create_app(self) -> Starlette:
        """Create the Starlette application."""
        verifier = OKBTokenVerifier(self._get_db_url)

        async def handle_sse(request: Request) -> Response:
            """Handle SSE connections for MCP."""
            # Verify token
            token = extract_token(request)
            if not token:
                return JSONResponse(
                    {"error": "Missing token. Use Authorization header or ?token= parameter"},
                    status_code=401,
                )

            token_info = verifier.verify(token)
            if not token_info:
                return JSONResponse(
                    {"error": "Invalid or expired token"},
                    status_code=401,
                )

            # Track existing sessions before connecting
            existing_sessions = set(self.transport._read_stream_writers.keys())

            async with self.transport.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                # Find the new session ID by comparing before/after
                current_sessions = set(self.transport._read_stream_writers.keys())
                new_sessions = current_sessions - existing_sessions
                if not new_sessions:
                    return JSONResponse(
                        {"error": "Failed to establish session"},
                        status_code=500,
                    )
                session_id = new_sessions.pop()
                session_id_hex = session_id.hex

                # Store token mapping for this session
                self.session_tokens[session_id_hex] = token_info
                self.server._current_token_info = token_info

                try:
                    await self.server.run(
                        read_stream, write_stream, self.server.create_initialization_options()
                    )
                finally:
                    # Clean up session on disconnect
                    self.session_tokens.pop(session_id_hex, None)

            return Response()

        async def handle_messages(scope, receive, send):
            """Handle POST messages for MCP (raw ASGI handler)."""
            request = Request(scope, receive)

            # Look up session from query params
            session_id = request.query_params.get("session_id")
            if not session_id:
                response = JSONResponse({"error": "Missing session_id"}, status_code=400)
                await response(scope, receive, send)
                return

            token_info = self.session_tokens.get(session_id)
            if not token_info:
                response = JSONResponse({"error": "Invalid or expired session"}, status_code=401)
                await response(scope, receive, send)
                return

            # Set current token info for tool calls
            self.server._current_token_info = token_info

            await self.transport.handle_post_message(scope, receive, send)

        async def health(request: Request) -> JSONResponse:
            """Health check endpoint."""
            return JSONResponse({"status": "ok"})

        routes = [
            Route("/health", health, methods=["GET"]),
            Route("/sse", handle_sse, methods=["GET"]),
            Mount("/messages", app=handle_messages),
        ]

        return Starlette(routes=routes)


def run_http_server(host: str = "127.0.0.1", port: int = 8080):
    """Run the HTTP MCP server."""
    import uvicorn

    print("Warming up embedding model...", file=sys.stderr)
    warmup()
    print("Ready.", file=sys.stderr)

    http_server = HTTPMCPServer()
    app = http_server.create_app()

    print(f"Starting HTTP MCP server on http://{host}:{port}", file=sys.stderr)
    print("  SSE endpoint: /sse", file=sys.stderr)
    print("  Messages endpoint: /messages/", file=sys.stderr)
    print("  Health endpoint: /health", file=sys.stderr)

    uvicorn.run(app, host=host, port=port, log_level="info")
