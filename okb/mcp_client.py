"""Thin MCP client for connecting to an OKB HTTP server."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class OKBClient:
    """MCP client that connects to an OKB HTTP server."""

    def __init__(self, server_url: str, token: str):
        self.server_url = server_url
        self.token = token

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call an MCP tool and return the text result."""
        http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(60, read=300),
        )
        async with http_client:
            async with streamable_http_client(
                url=self.server_url,
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments or {})
                    # Collect text from all content blocks
                    parts = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            parts.append(block.text)
                    return "\n".join(parts)

    def call_tool_sync(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Synchronous wrapper around call_tool."""
        return asyncio.run(self.call_tool(name, arguments))
