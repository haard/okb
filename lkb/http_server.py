"""HTTP transport server for MCP with Bearer token authentication.

This module provides an HTTP server that serves the LKB MCP server with
token-based authentication. A single HTTP server can serve multiple databases,
with the token determining which database to use.
"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import CallToolResult, TextContent, Tool
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .config import config
from .local_embedder import warmup
from .mcp_server import (
    KnowledgeBase,
    format_actionable_items,
    format_search_results,
)
from .tokens import LKBTokenVerifier, TokenInfo

# Permission sets
READ_ONLY_TOOLS = frozenset(
    {
        "search_knowledge",
        "keyword_search",
        "hybrid_search",
        "get_document",
        "list_sources",
        "list_projects",
        "recent_documents",
        "get_actionable_items",
    }
)

WRITE_TOOLS = frozenset(
    {
        "save_knowledge",
        "delete_knowledge",
    }
)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to verify Bearer tokens and attach auth context to requests."""

    def __init__(self, app, verifier: LKBTokenVerifier):
        super().__init__(app)
        self.verifier = verifier

    async def dispatch(self, request: Request, call_next):
        # Allow health check without auth
        if request.url.path == "/health":
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )

        token = auth_header[7:]  # Remove 'Bearer ' prefix
        token_info = self.verifier.verify(token)

        if not token_info:
            return JSONResponse(
                {"error": "Invalid or expired token"},
                status_code=401,
            )

        # Attach token info to request state
        request.state.token_info = token_info

        return await call_next(request)


class HTTPMCPServer:
    """HTTP server for MCP with token authentication."""

    def __init__(self):
        self.knowledge_bases: dict[str, KnowledgeBase] = {}
        self.server = Server("knowledge-base")
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

            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                )

        except Exception as e:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: {e!s}")])

    def create_app(self) -> Starlette:
        """Create the Starlette application with auth middleware."""
        verifier = LKBTokenVerifier(self._get_db_url)

        async def handle_sse(request: Request) -> Response:
            """Handle SSE connections for MCP."""
            # Attach token info to server for tool calls
            self.server._current_token_info = request.state.token_info

            transport = SseServerTransport("/messages/")
            async with transport.connect_sse(request.scope, request.receive, request._send) as (
                read_stream,
                write_stream,
            ):
                await self.server.run(
                    read_stream, write_stream, self.server.create_initialization_options()
                )

            return Response()

        async def handle_messages(request: Request) -> Response:
            """Handle POST messages for MCP."""
            # Attach token info to server for tool calls
            self.server._current_token_info = request.state.token_info

            transport = SseServerTransport("/messages/")
            return await transport.handle_post_message(
                request.scope, request.receive, request._send
            )

        async def health(request: Request) -> JSONResponse:
            """Health check endpoint."""
            return JSONResponse({"status": "ok"})

        routes = [
            Route("/health", health, methods=["GET"]),
            Route("/sse", handle_sse, methods=["GET"]),
            Route("/messages/", handle_messages, methods=["POST"]),
        ]

        middleware = [
            Middleware(TokenAuthMiddleware, verifier=verifier),
        ]

        return Starlette(routes=routes, middleware=middleware)


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
