"""Background sync runner for non-blocking trigger_sync / trigger_rescan.

Spawns asyncio tasks that run syncs in background threads, updating sync_state
status columns so callers can poll via list_sync_sources.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from functools import partial
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import config

log = logging.getLogger(__name__)

# In-process tracking of running tasks: (source_name, db_name) -> Task
_running: dict[tuple[str, str], asyncio.Task] = {}

RESCAN_SOURCE = "_rescan"


def _set_status(
    db_url: str,
    source_name: str,
    db_name: str,
    status: str,
    error: str | None = None,
    started_at: datetime | None = None,
) -> None:
    """Update status fields in sync_state."""
    with psycopg.connect(db_url) as conn:
        conn.execute(
            """INSERT INTO sync_state (source_name, database_name, status, started_at, error)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (source_name, database_name)
               DO UPDATE SET status = EXCLUDED.status,
                            started_at = EXCLUDED.started_at,
                            error = EXCLUDED.error""",
            (source_name, db_name, status, started_at, error),
        )
        conn.commit()


def _run_sync_single(
    db_url: str,
    source_name: str,
    db_name: str,
    full: bool = False,
    source_overrides: dict[str, Any] | None = None,
) -> str:
    """Sync a single source. Returns result string.

    This is the extracted inner loop from _run_sync in mcp_server.py.
    """
    from .ingest import Ingester
    from .plugins.registry import PluginRegistry

    source = PluginRegistry.get_source(source_name)
    if source is None:
        return f"{source_name}: not found"

    source_cfg = config.get_source_config(source_name, db_name)
    if source_cfg is None:
        return f"{source_name}: not configured or disabled"

    if source_overrides:
        source_cfg = {**source_cfg, **source_overrides}

    try:
        source.configure(source_cfg)
    except Exception as e:
        return f"{source_name}: config error - {e}"

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        from .mcp_server import _get_sync_state, _save_sync_state

        state = None if full else _get_sync_state(conn, source_name, db_name)

        try:
            documents, new_state = source.fetch(state)
        except Exception as e:
            return f"{source_name}: fetch error - {e}"

        if documents:
            ingester = Ingester(db_url, use_modal=True)
            ingester.ingest_documents(documents)
            result = f"{source_name}: synced {len(documents)} documents"
        else:
            result = f"{source_name}: no new documents"

        _save_sync_state(conn, source_name, db_name, new_state)

    return result


async def _run_source_task(
    db_url: str,
    source_name: str,
    db_name: str,
    full: bool,
    source_overrides: dict[str, Any] | None,
) -> None:
    """Background task wrapper: sets status, runs sync, clears status."""
    key = (source_name, db_name)
    try:
        _set_status(db_url, source_name, db_name, "running", started_at=datetime.now(UTC))
        result = await asyncio.to_thread(
            partial(
                _run_sync_single,
                db_url,
                source_name,
                db_name,
                full=full,
                source_overrides=source_overrides,
            )
        )
        log.info("sync %s/%s: %s", source_name, db_name, result)

        # Check for errors in the result string
        if "error" in result:
            _set_status(db_url, source_name, db_name, "error", error=result)
        else:
            _set_status(db_url, source_name, db_name, "idle")
    except Exception as e:
        log.exception("sync %s/%s failed", source_name, db_name)
        _set_status(db_url, source_name, db_name, "error", error=str(e))
    finally:
        _running.pop(key, None)


def _build_source_overrides(
    source_name: str,
    *,
    doc_ids: list[str] | None = None,
    repos: list[str] | None = None,
    branch: str | None = None,
    include_issues: bool = False,
    include_prs: bool = False,
    include_wiki: bool = False,
    include_source: bool = False,
    folders: list[str] | None = None,
    channels: list[str] | None = None,
) -> dict[str, Any]:
    """Build source-specific config overrides from tool arguments."""
    overrides: dict[str, Any] = {}
    if doc_ids:
        overrides["doc_ids"] = doc_ids
    if repos:
        overrides["repos"] = repos
    if branch:
        overrides["branch"] = branch
    if include_issues:
        overrides["include_issues"] = True
    if include_prs:
        overrides["include_prs"] = True
    if include_wiki:
        overrides["include_wiki"] = True
    if include_source:
        overrides["include_source"] = True
    if folders:
        overrides["folders"] = folders
    if channels:
        overrides["channels"] = channels
    return overrides


async def start_sync(
    db_url: str,
    db_name: str,
    sources: list[str],
    sync_all: bool = False,
    full: bool = False,
    **kwargs,
) -> str:
    """Start background sync for the given sources. Returns immediately.

    kwargs are passed through as source overrides (repos, branch, etc.)
    """
    from .plugins.registry import PluginRegistry

    if db_name is None:
        db_name = config.get_database().name

    # Determine which sources to sync
    if sync_all:
        source_names = config.list_enabled_sources(db_name)
    elif sources:
        source_names = list(sources)
    else:
        # No sources specified — return available list (same as before)
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

    started = []
    already_running = []

    for source_name in source_names:
        key = (source_name, db_name)
        if key in _running and not _running[key].done():
            already_running.append(source_name)
            continue

        overrides = _build_source_overrides(source_name, **kwargs)
        task = asyncio.create_task(
            _run_source_task(db_url, source_name, db_name, full, overrides)
        )
        _running[key] = task
        started.append(source_name)

    parts = []
    if started:
        parts.append(f"Sync started for: {', '.join(started)}")
    if already_running:
        parts.append(f"Already running: {', '.join(already_running)}")
    parts.append("Use list_sync_sources to check progress.")
    return "\n".join(parts)


async def _run_rescan_task(
    db_url: str,
    db_name: str,
    dry_run: bool,
    delete_missing: bool,
) -> None:
    """Background task wrapper for rescan."""
    key = (RESCAN_SOURCE, db_name)
    try:
        _set_status(db_url, RESCAN_SOURCE, db_name, "running",
                     started_at=datetime.now(UTC))
        from .mcp_server import _run_rescan

        result = await asyncio.to_thread(
            partial(_run_rescan, db_url, dry_run=dry_run, delete_missing=delete_missing)
        )
        log.info("rescan %s: %s", db_name, result)
        _set_status(db_url, RESCAN_SOURCE, db_name, "idle")
    except Exception as e:
        log.exception("rescan %s failed", db_name)
        _set_status(db_url, RESCAN_SOURCE, db_name, "error", error=str(e))
    finally:
        _running.pop(key, None)


async def start_rescan(
    db_url: str,
    db_name: str,
    dry_run: bool = False,
    delete_missing: bool = False,
) -> str:
    """Start background rescan. Returns immediately (unless dry_run)."""
    if db_name is None:
        db_name = config.get_database().name

    # Dry run is fast and the caller wants the result — run synchronously
    if dry_run:
        from .mcp_server import _run_rescan

        return await asyncio.to_thread(
            partial(_run_rescan, db_url, dry_run=True, delete_missing=delete_missing)
        )

    key = (RESCAN_SOURCE, db_name)
    if key in _running and not _running[key].done():
        return "Rescan already running. Use list_sync_sources to check progress."

    task = asyncio.create_task(
        _run_rescan_task(db_url, db_name, dry_run, delete_missing)
    )
    _running[key] = task
    return "Rescan started. Use list_sync_sources to check progress."
