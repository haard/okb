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

import asyncio
import sys
import time
from typing import Any

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent, Tool
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import config
from .local_embedder import warmup
from .mcp_server import KnowledgeBase
from .tokens import OKBTokenVerifier, TokenInfo
from .tools import execute_tool

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
        "ingest_documents",
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
        from . import __version__

        self.knowledge_bases: dict[str, KnowledgeBase] = {}
        self.server = Server(
            "knowledge-base", version=__version__, instructions=f"OKB v{__version__}"
        )
        # Session manager handles all transport complexity
        self.session_manager = StreamableHTTPSessionManager(app=self.server)
        # Map mcp-session-id -> (token_info, created_monotonic)
        self.session_tokens: dict[str, tuple[TokenInfo, float]] = {}
        self._max_sessions = 1000
        self._session_ttl = 86400  # 24 hours
        # Per-KB asyncio locks for connection safety
        self.kb_locks: dict[str, asyncio.Lock] = {}
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
            # Look up auth context via the MCP request_context (per-task)
            token_info: TokenInfo | None = None
            try:
                req = self.server.request_context.request
                if req is not None:
                    session_id = req.headers.get("mcp-session-id")
                    if session_id:
                        entry = self.session_tokens.get(session_id)
                        if entry is not None:
                            token_info = entry[0]
            except LookupError:
                pass

            if token_info is None:
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: No authentication context")]
                )

            # Check permissions (allow-list: reject unknown tools for ro tokens)
            if token_info.permissions == "ro" and name not in READ_ONLY_TOOLS:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Error: Permission denied. Tool '{name}' requires write access.",
                        )
                    ]
                )

            # Get or create knowledge base and lock for this database
            db = token_info.database
            if db not in self.knowledge_bases:
                db_url = self._get_db_url(db)
                self.knowledge_bases[db] = KnowledgeBase(db_url)
            if db not in self.kb_locks:
                self.kb_locks[db] = asyncio.Lock()

            kb = self.knowledge_bases[db]
            kb_lock = self.kb_locks[db]

            # Execute the tool
            db_cfg = config.get_database(db)
            return await execute_tool(
                kb, name, arguments, db_name=db, db_config=db_cfg, kb_lock=kb_lock
            )

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
                    existing_entry = self.session_tokens.get(session_id)
                    if existing_entry:
                        existing_token, created_at = existing_entry
                        # Evict expired sessions on access
                        if time.monotonic() - created_at > self._session_ttl:
                            self.session_tokens.pop(session_id, None)
                            existing_token = None
                    else:
                        existing_token = None
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
                                    # Evict oldest sessions if at capacity
                                    if len(self.session_tokens) >= self._max_sessions:
                                        now = time.monotonic()
                                        # Remove expired first
                                        expired = [
                                            k
                                            for k, (_, t) in self.session_tokens.items()
                                            if now - t > self._session_ttl
                                        ]
                                        for k in expired:
                                            del self.session_tokens[k]
                                        # If still full, remove oldest
                                        if len(self.session_tokens) >= self._max_sessions:
                                            oldest = min(
                                                self.session_tokens,
                                                key=lambda k: self.session_tokens[k][1],
                                            )
                                            del self.session_tokens[oldest]
                                    self.session_tokens[captured_session_id] = (
                                        token_info,
                                        time.monotonic(),
                                    )
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
