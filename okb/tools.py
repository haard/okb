"""Shared tool dispatch for MCP servers.

Contains the canonical execute_tool() function used by both the stdio
(mcp_server.py) and HTTP (http_server.py) transports, plus formatting helpers.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, TextContent

if TYPE_CHECKING:
    from .config import DatabaseConfig
    from .mcp_server import KnowledgeBase


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


async def _run_with_lock(fn, *args, lock: asyncio.Lock | None = None, **kwargs):
    """Run a sync function, optionally under an asyncio lock and in a thread.

    The lock serializes access to the shared KnowledgeBase connection (psycopg
    connections are not thread-safe).  ``to_thread`` offloads the blocking I/O
    so the event loop stays responsive.

    When lock is None (stdio transport), the function runs directly with zero overhead.
    """
    if lock is not None:
        async with lock:
            return await asyncio.to_thread(fn, *args, **kwargs)
    return fn(*args, **kwargs)


async def execute_tool(
    kb: KnowledgeBase,
    name: str,
    arguments: dict[str, Any],
    *,
    db_name: str | None = None,
    db_config: DatabaseConfig | None = None,
    kb_lock: asyncio.Lock | None = None,
) -> CallToolResult:
    """Execute an MCP tool on a knowledge base.

    This is the single canonical dispatch for all MCP tools, used by both
    the stdio and HTTP transports.

    Args:
        kb: The KnowledgeBase instance to operate on.
        name: Tool name.
        arguments: Tool arguments dict.
        db_name: Database name (for sync state, snapshots, etc.).
        db_config: Database config (for get_database_info).
        kb_lock: Per-KB asyncio lock for HTTP transport (None for stdio).
    """
    try:
        if name == "search_knowledge":
            results = await _run_with_lock(
                kb.semantic_search,
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type=arguments.get("source_type"),
                project=arguments.get("project"),
                since=arguments.get("since"),
                lock=kb_lock,
            )
            return CallToolResult(
                content=[TextContent(type="text", text=format_search_results(results))]
            )

        elif name == "keyword_search":
            results = await _run_with_lock(
                kb.keyword_search,
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type=arguments.get("source_type"),
                since=arguments.get("since"),
                lock=kb_lock,
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=format_search_results(results, show_similarity=False)
                    )
                ]
            )

        elif name == "hybrid_search":
            results = await _run_with_lock(
                kb.hybrid_search,
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type=arguments.get("source_type"),
                since=arguments.get("since"),
                lock=kb_lock,
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=format_search_results(results, show_similarity=False)
                    )
                ]
            )

        elif name == "get_document":
            doc = await _run_with_lock(kb.get_document, arguments["source_path"], lock=kb_lock)
            if not doc:
                return CallToolResult(
                    content=[TextContent(type="text", text="Document not found.")]
                )
            return CallToolResult(
                content=[TextContent(type="text", text=f"# {doc['title']}\n\n{doc['content']}")]
            )

        elif name == "list_sources":
            sources = await _run_with_lock(kb.list_sources, lock=kb_lock)
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
            projects = await _run_with_lock(kb.list_projects, lock=kb_lock)
            if not projects:
                return CallToolResult(content=[TextContent(type="text", text="No projects found.")])
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text="## Projects\n\n" + "\n".join(f"- {p}" for p in projects)
                    )
                ]
            )

        elif name == "get_project_stats":
            stats = await _run_with_lock(kb.get_project_stats, lock=kb_lock)
            if not stats:
                return CallToolResult(content=[TextContent(type="text", text="No projects found.")])
            output = ["## Project Statistics\n"]
            output.append("| Project | Documents | Source Types |")
            output.append("|---------|-----------|--------------|")
            for s in stats:
                types = ", ".join(s["source_types"]) if s["source_types"] else "-"
                output.append(f"| {s['project']} | {s['doc_count']} | {types} |")
            return CallToolResult(content=[TextContent(type="text", text="\n".join(output))])

        elif name == "list_documents_by_project":
            project = arguments["project"]
            limit = arguments.get("limit", 100)
            docs = await _run_with_lock(
                kb.list_documents_by_project, project, limit, lock=kb_lock
            )
            if not docs:
                msg = f"No documents found for project '{project}'."
                return CallToolResult(content=[TextContent(type="text", text=msg)])
            output = [f"## Documents in '{project}' ({len(docs)} documents)\n"]
            for d in docs:
                output.append(f"- **{d['title'] or d['source_path']}** ({d['source_type']})")
                output.append(f"  - `{d['source_path']}`")
            return CallToolResult(content=[TextContent(type="text", text="\n".join(output))])

        elif name == "rename_project":
            old_name = arguments["old_name"]
            new_name = arguments["new_name"]
            if old_name == new_name:
                return CallToolResult(
                    content=[TextContent(type="text", text="Old and new names are the same.")]
                )
            count = await _run_with_lock(kb.rename_project, old_name, new_name, lock=kb_lock)
            if count == 0:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"No documents found with project '{old_name}'.",
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
            success = await _run_with_lock(
                kb.set_document_project, source_path, project, lock=kb_lock
            )
            if not success:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Document not found: {source_path}")]
                )
            if project:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Set project to '{project}' for {source_path}",
                        )
                    ]
                )
            return CallToolResult(
                content=[TextContent(type="text", text=f"Cleared project for {source_path}")]
            )

        elif name == "recent_documents":
            docs = await _run_with_lock(
                kb.get_recent_documents, arguments.get("limit", 10), lock=kb_lock
            )
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
            result = await _run_with_lock(
                kb.save_knowledge,
                title=arguments["title"],
                content=arguments["content"],
                tags=arguments.get("tags"),
                project=arguments.get("project"),
                source_type=arguments.get("source_type", "claude-note"),
                lock=kb_lock,
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
            deleted = await _run_with_lock(
                kb.delete_knowledge, arguments["source_path"], lock=kb_lock
            )
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

        elif name == "update_knowledge":
            result = await _run_with_lock(
                kb.update_knowledge,
                source_path=arguments["source_path"],
                title=arguments.get("title"),
                content=arguments.get("content"),
                tags=arguments.get("tags"),
                project=arguments.get("project"),
                lock=kb_lock,
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

        elif name == "get_actionable_items":
            items = await _run_with_lock(
                kb.get_actionable_items,
                item_type=arguments.get("item_type"),
                status=arguments.get("status"),
                due_date=arguments.get("due_date"),
                event_date=arguments.get("event_date"),
                min_priority=arguments.get("min_priority"),
                limit=arguments.get("limit", 20),
                lock=kb_lock,
            )
            return CallToolResult(
                content=[TextContent(type="text", text=format_actionable_items(items))]
            )

        elif name == "get_database_info":
            info_parts = ["## Knowledge Base Info\n"]

            # Config-defined description/topics
            if db_config and db_config.description:
                info_parts.append(f"**Description (config):** {db_config.description}")
            if db_config and db_config.topics:
                info_parts.append(f"**Topics (config):** {', '.join(db_config.topics)}")

            # LLM-enhanced metadata
            llm_desc = None
            llm_topics = None
            try:
                metadata = await _run_with_lock(kb.get_database_metadata, lock=kb_lock)
                llm_desc = metadata.get("llm_description", {}).get("value")
                llm_topics = metadata.get("llm_topics", {}).get("value")
                if llm_desc:
                    info_parts.append(f"**Description (LLM-enhanced):** {llm_desc}")
                if llm_topics:
                    info_parts.append(f"**Topics (LLM-enhanced):** {', '.join(llm_topics)}")
            except Exception:
                pass  # Table may not exist yet

            # Content stats
            sources = await _run_with_lock(kb.list_sources, lock=kb_lock)
            if sources:
                info_parts.append("\n### Content Statistics")
                for s in sources:
                    tokens = s.get("total_tokens") or 0
                    info_parts.append(
                        f"- **{s['source_type']}**: {s['document_count']} documents, "
                        f"{s['chunk_count']} chunks (~{tokens:,} tokens)"
                    )

            # Projects
            projects = await _run_with_lock(kb.list_projects, lock=kb_lock)
            if projects:
                info_parts.append(f"\n### Projects\n{', '.join(projects)}")

            # Hint if no description exists
            config_desc = db_config.description if db_config else None
            if not config_desc and not llm_desc:
                info_parts.append(
                    "\n**Note:** No description set. Consider exploring with "
                    "recent_documents and search_knowledge, then calling "
                    "set_database_description to document what's here."
                )

            return CallToolResult(content=[TextContent(type="text", text="\n".join(info_parts))])

        elif name == "set_database_description":
            updated = []
            if "description" in arguments:
                await _run_with_lock(
                    kb.set_database_metadata, "llm_description", arguments["description"],
                    lock=kb_lock,
                )
                updated.append("description")
            if "topics" in arguments:
                await _run_with_lock(
                    kb.set_database_metadata, "llm_topics", arguments["topics"],
                    lock=kb_lock,
                )
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
            result = await _run_with_lock(
                kb.save_todo,
                title=arguments["title"],
                content=arguments.get("content"),
                due_date=arguments.get("due_date"),
                priority=arguments.get("priority"),
                project=arguments.get("project"),
                tags=arguments.get("tags"),
                lock=kb_lock,
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
            from .sync_runner import start_sync

            result = await start_sync(
                kb.db_url,
                db_name or "default",
                sources=arguments.get("sources", []),
                sync_all=arguments.get("all", False),
                full=arguments.get("full", False),
                doc_ids=arguments.get("doc_ids"),
                repos=arguments.get("repos"),
                branch=arguments.get("branch"),
                include_issues=arguments.get("include_issues", False),
                include_prs=arguments.get("include_prs", False),
                include_wiki=arguments.get("include_wiki", False),
                include_source=arguments.get("include_source", False),
                folders=arguments.get("folders"),
                channels=arguments.get("channels"),
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "ingest_documents":
            for doc in arguments.get("documents", []):
                sp = doc.get("source_path", "")
                if sp.startswith("claude://") or sp.startswith("okb://"):
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error: source_path '{sp}' is an internal path. "
                                "Use save_knowledge for LLM-generated content; "
                                "ingest_documents is for external sources (URLs, file paths).",
                            )
                        ]
                    )

            from .mcp_server import _run_ingest

            result = await asyncio.to_thread(
                partial(
                    _run_ingest,
                    kb.db_url,
                    arguments["documents"],
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "trigger_rescan":
            from .sync_runner import start_rescan

            result = await start_rescan(
                kb.db_url,
                db_name or "default",
                dry_run=arguments.get("dry_run", False),
                delete_missing=arguments.get("delete_missing", False),
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_sync_sources":
            from .mcp_server import _list_sync_sources

            # These helpers create their own DB connections, so the KB lock
            # isn't needed — to_thread is purely for event loop responsiveness.
            _fn = partial(_list_sync_sources, kb.db_url, db_name or "default")
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "get_synthesis_samples":
            from .mcp_server import _get_synthesis_samples

            _fn = partial(
                _get_synthesis_samples,
                kb.db_url,
                project=arguments.get("project"),
                sample_size=arguments.get("sample_size", 20),
                strategy=arguments.get("strategy", "diverse"),
                excerpt_length=arguments.get("excerpt_length", 1500),
            )
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "synthesize_knowledge":
            from .mcp_server import _synthesize_knowledge

            result = await asyncio.to_thread(
                partial(
                    _synthesize_knowledge,
                    kb.db_url,
                    project=arguments.get("project"),
                    sample_size=arguments.get("sample_size", 25),
                    max_proposals=arguments.get("max_proposals", 10),
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_pending_synthesis":
            from .mcp_server import _list_pending_synthesis

            _fn = partial(
                _list_pending_synthesis,
                kb.db_url,
                project=arguments.get("project"),
                limit=arguments.get("limit", 50),
            )
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "approve_synthesis":
            from .mcp_server import _approve_synthesis

            _fn = partial(
                _approve_synthesis,
                kb.db_url,
                arguments["pending_id"],
                title=arguments.get("title"),
                content=arguments.get("content"),
            )
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "reject_synthesis":
            from .mcp_server import _reject_synthesis

            _fn = partial(_reject_synthesis, kb.db_url, arguments["pending_id"])
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "edit_pending_synthesis":
            from .mcp_server import _edit_pending_synthesis

            _fn = partial(
                _edit_pending_synthesis,
                kb.db_url,
                arguments["pending_id"],
                title=arguments.get("title"),
                content=arguments.get("content"),
            )
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "analyze_knowledge_base":
            from .mcp_server import _analyze_knowledge_base

            result = await asyncio.to_thread(
                partial(
                    _analyze_knowledge_base,
                    kb.db_url,
                    project=arguments.get("project"),
                    sample_size=arguments.get("sample_size", 15),
                    auto_update=arguments.get("auto_update", True),
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "save_snapshot":
            from .mcp_server import _save_snapshot

            _fn = partial(_save_snapshot, arguments.get("name"), db_name=db_name)
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_snapshots":
            from .mcp_server import _list_snapshots

            _fn = partial(_list_snapshots, db_name=db_name)
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "restore_snapshot":
            from .mcp_server import _restore_snapshot

            _fn = partial(
                _restore_snapshot,
                arguments["name"],
                arguments.get("confirm", False),
                db_name=db_name,
            )
            result = await asyncio.to_thread(_fn) if kb_lock is not None else _fn()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])

    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Error: {e!s}")])
