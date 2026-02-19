"""HTTP transport server for MCP with token authentication.

This module provides an HTTP server that serves the OKB MCP server with
token-based authentication using Streamable HTTP transport. Tokens can be
passed via Authorization header or query parameter. A single HTTP server
can serve multiple databases, with the token determining which database to use.

Transport: Streamable HTTP (RFC 9728 compliant)
- POST /mcp → send JSON-RPC messages, get SSE response
- GET /mcp → optional standalone SSE for server notifications
- DELETE /mcp → terminate session
- Session ID in Mcp-Session-Id header
"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent, Tool
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

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
        "get_project_stats",
        "recent_documents",
        "get_actionable_items",
        "get_database_info",
        "list_sync_sources",
        "get_synthesis_samples",
        "list_pending_synthesis",
        "list_snapshots",
    }
)

WRITE_TOOLS = frozenset(
    {
        "save_knowledge",
        "delete_knowledge",
        "update_knowledge",
        "set_database_description",
        "add_todo",
        "trigger_sync",
        "trigger_rescan",
        "synthesize_knowledge",
        "approve_synthesis",
        "reject_synthesis",
        "edit_pending_synthesis",
        "analyze_knowledge_base",
        "rename_project",
        "set_document_project",
        "save_snapshot",
        "restore_snapshot",
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
    """HTTP server for MCP with token authentication using Streamable HTTP transport."""

    def __init__(self):
        self.knowledge_bases: dict[str, KnowledgeBase] = {}
        self.server = Server("knowledge-base")
        # Session manager handles all transport complexity
        self.session_manager = StreamableHTTPSessionManager(app=self.server)
        # Map mcp-session-id -> token_info
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

            elif name == "get_project_stats":
                stats = kb.get_project_stats()
                if not stats:
                    return CallToolResult(
                        content=[TextContent(type="text", text="No projects found.")]
                    )
                output = ["## Project Statistics\n"]
                for p in stats:
                    output.append(f"- **{p['project']}**: {p['document_count']} documents")
                return CallToolResult(content=[TextContent(type="text", text="\n".join(output))])

            elif name == "rename_project":
                old_name = arguments["old_name"]
                new_name = arguments["new_name"]
                if old_name == new_name:
                    return CallToolResult(
                        content=[TextContent(type="text", text="Old and new names are the same.")]
                    )
                count = kb.rename_project(old_name, new_name)
                if count == 0:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text", text=f"No documents found with project '{old_name}'."
                            )
                        ]
                    )
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Renamed project '{old_name}' to '{new_name}' "
                            f"({count} documents updated).",
                        )
                    ]
                )

            elif name == "set_document_project":
                source_path = arguments["source_path"]
                project = arguments.get("project")
                success = kb.set_document_project(source_path, project)
                if not success:
                    return CallToolResult(
                        content=[
                            TextContent(type="text", text=f"Document not found: {source_path}")
                        ]
                    )
                if project:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text", text=f"Set project to '{project}' for {source_path}"
                            )
                        ]
                    )
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Cleared project for {source_path}")]
                )

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
                    source_type=arguments.get("source_type", "claude-note"),
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

            elif name == "get_synthesis_samples":
                from .mcp_server import _get_synthesis_samples

                result = _get_synthesis_samples(
                    kb.db_url,
                    project=arguments.get("project"),
                    sample_size=arguments.get("sample_size", 20),
                    strategy=arguments.get("strategy", "diverse"),
                    excerpt_length=arguments.get("excerpt_length", 1500),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "synthesize_knowledge":
                from .mcp_server import _synthesize_knowledge

                result = _synthesize_knowledge(
                    kb.db_url,
                    project=arguments.get("project"),
                    sample_size=arguments.get("sample_size", 25),
                    max_proposals=arguments.get("max_proposals", 10),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "list_pending_synthesis":
                from .mcp_server import _list_pending_synthesis

                result = _list_pending_synthesis(
                    kb.db_url,
                    project=arguments.get("project"),
                    limit=arguments.get("limit", 50),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "approve_synthesis":
                from .mcp_server import _approve_synthesis

                result = _approve_synthesis(
                    kb.db_url,
                    arguments["pending_id"],
                    title=arguments.get("title"),
                    content=arguments.get("content"),
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "reject_synthesis":
                from .mcp_server import _reject_synthesis

                result = _reject_synthesis(kb.db_url, arguments["pending_id"])
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "edit_pending_synthesis":
                from .mcp_server import _edit_pending_synthesis

                result = _edit_pending_synthesis(
                    kb.db_url,
                    arguments["pending_id"],
                    title=arguments.get("title"),
                    content=arguments.get("content"),
                )
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

            elif name == "update_knowledge":
                result = kb.update_knowledge(
                    source_path=arguments["source_path"],
                    title=arguments.get("title"),
                    content=arguments.get("content"),
                    tags=arguments.get("tags"),
                    project=arguments.get("project"),
                )
                if result["status"] == "not_found":
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Document not found: `{result['source_path']}`",
                            )
                        ]
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
                                f"Knowledge updated successfully:\n"
                                f"- Title: {result['title']}\n"
                                f"- Path: `{result['source_path']}`\n"
                                f"- Tokens: ~{result['token_count']}"
                            ),
                        )
                    ]
                )

            elif name == "save_snapshot":
                from .mcp_server import _save_snapshot

                token_info = getattr(self.server, "_current_token_info", None)
                db_name = token_info.database if token_info else None
                result = _save_snapshot(arguments.get("name"), db_name=db_name)
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "list_snapshots":
                from .mcp_server import _list_snapshots

                token_info = getattr(self.server, "_current_token_info", None)
                db_name = token_info.database if token_info else None
                result = _list_snapshots(db_name=db_name)
                return CallToolResult(content=[TextContent(type="text", text=result)])

            elif name == "restore_snapshot":
                from .mcp_server import _restore_snapshot

                token_info = getattr(self.server, "_current_token_info", None)
                db_name = token_info.database if token_info else None
                result = _restore_snapshot(
                    arguments["name"],
                    arguments.get("confirm", False),
                    db_name=db_name,
                )
                return CallToolResult(content=[TextContent(type="text", text=result)])

            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                )

        except Exception as e:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: {e!s}")])

    def create_app(self):
        """Create the Starlette application."""
        verifier = OKBTokenVerifier(self._get_db_url)
        session_header_name = "mcp-session-id"

        def create_mcp_handler():
            """Create an ASGI handler for MCP with auth."""

            async def handle_mcp(scope, receive, send):
                """Handle all MCP requests (GET, POST, DELETE) with auth."""
                request = Request(scope, receive)

                # Extract and verify token
                token = extract_token(request)
                if not token:
                    response = JSONResponse(
                        {"error": "Missing token. Use Authorization header or ?token= param"},
                        status_code=401,
                    )
                    await response(scope, receive, send)
                    return

                token_info = verifier.verify(token)
                if not token_info:
                    response = JSONResponse(
                        {"error": "Invalid or expired token"},
                        status_code=401,
                    )
                    await response(scope, receive, send)
                    return

                # Check if this is an existing session
                session_id = request.headers.get(session_header_name)
                if session_id:
                    # Verify token matches existing session (compare by hash and db, not object)
                    existing_token = self.session_tokens.get(session_id)
                    if existing_token:
                        # Token must match the one used to create the session
                        if (
                            existing_token.token_hash != token_info.token_hash
                            or existing_token.database != token_info.database
                        ):
                            response = JSONResponse(
                                {"error": "Token mismatch for session"},
                                status_code=401,
                            )
                            await response(scope, receive, send)
                            return

                # Set current token info for tool calls
                self.server._current_token_info = token_info

                # Wrap send to capture the session ID from response headers
                captured_session_id = None

                async def send_wrapper(message):
                    nonlocal captured_session_id
                    if message["type"] == "http.response.start":
                        headers = message.get("headers", [])
                        for name, value in headers:
                            header_name = (
                                name.lower() if isinstance(name, bytes) else name.lower().encode()
                            )
                            if header_name == session_header_name.encode():
                                captured_session_id = (
                                    value.decode() if isinstance(value, bytes) else value
                                )
                                # Store immediately since SSE keeps connection open
                                if captured_session_id not in self.session_tokens:
                                    self.session_tokens[captured_session_id] = token_info
                                break
                    await send(message)

                # Delegate to session manager
                await self.session_manager.handle_request(scope, receive, send_wrapper)

            return handle_mcp

        # Create the MCP handler ASGI app
        mcp_handler = create_mcp_handler()

        # Custom ASGI app that routes /mcp and /sse to MCP handler
        async def router(scope, receive, send):
            if scope["type"] == "http":
                path = scope["path"].rstrip("/")  # Handle trailing slash
                if path in ("/mcp", "/sse"):
                    await mcp_handler(scope, receive, send)
                    return
                elif path == "/health" or scope["path"] == "/health":
                    response = JSONResponse({"status": "ok"})
                    await response(scope, receive, send)
                    return
            # 404 for unknown paths
            response = JSONResponse({"error": "Not found"}, status_code=404)
            await response(scope, receive, send)

        # Wrap with lifespan handling
        async def app_with_lifespan(scope, receive, send):
            if scope["type"] == "lifespan":
                async with self.session_manager.run():
                    # Handle lifespan protocol
                    while True:
                        message = await receive()
                        if message["type"] == "lifespan.startup":
                            await send({"type": "lifespan.startup.complete"})
                        elif message["type"] == "lifespan.shutdown":
                            await send({"type": "lifespan.shutdown.complete"})
                            return
            else:
                await router(scope, receive, send)

        # Add CORS for browser clients - wrap the raw ASGI app
        app = CORSMiddleware(
            app_with_lifespan,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", session_header_name],
            expose_headers=[session_header_name],
        )

        return app


def run_http_server(host: str = "127.0.0.1", port: int = 8080):
    """Run the HTTP MCP server with Streamable HTTP transport."""
    import uvicorn

    print("Warming up embedding model...", file=sys.stderr)
    warmup()
    print("Ready.", file=sys.stderr)

    http_server = HTTPMCPServer()
    app = http_server.create_app()

    print(f"Starting HTTP MCP server on http://{host}:{port}", file=sys.stderr)
    print("  MCP endpoint: /mcp (GET, POST, DELETE)", file=sys.stderr)
    print("  MCP endpoint: /sse (alias for /mcp)", file=sys.stderr)
    print("  Health endpoint: /health", file=sys.stderr)
    print("  Transport: Streamable HTTP", file=sys.stderr)

    uvicorn.run(app, host=host, port=port, log_level="info")
