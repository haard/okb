"""CLI client that operates against a remote OKB MCP server."""

from __future__ import annotations

import sys

import click

from .config import config
from .mcp_client import OKBClient


def _get_client(server_name: str | None = None) -> OKBClient:
    """Get an MCP client from config (env vars > config file)."""
    try:
        sc = config.get_server(server_name)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    if not sc.url:
        click.echo(
            f"No URL configured for server '{sc.name}'. "
            "Set url in servers: config or OKB_SERVER_URL.",
            err=True,
        )
        sys.exit(1)
    if not sc.token:
        click.echo(
            f"No token configured for server '{sc.name}'. "
            "Set token in servers: config or OKB_TOKEN.",
            err=True,
        )
        sys.exit(1)
    return OKBClient(sc.url, sc.token)


def _call(name: str, arguments: dict | None = None) -> None:
    """Call a tool and print the result."""
    ctx = click.get_current_context()
    server_name = ctx.obj.get("server") if ctx.obj else None
    client = _get_client(server_name)
    try:
        result = client.call_tool_sync(name, arguments)
        click.echo(result)
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.group()
@click.version_option(package_name="okb")
@click.option("-s", "--server", default=None, help="Named server from config")
@click.pass_context
def main(ctx, server):
    """OKB client - query a remote knowledge base via MCP."""
    ctx.ensure_object(dict)
    ctx.obj["server"] = server


# ---------------------------------------------------------------------------
# Search commands
# ---------------------------------------------------------------------------


@main.command()
@click.argument("query")
@click.option("-n", "--limit", default=5, type=int, help="Max results")
@click.option("--type", "source_type", default=None, help="Filter by source type")
@click.option("--since", default=None, help="Time filter (e.g. 7d, 30d, 6mo)")
def search(query: str, limit: int, source_type: str | None, since: str | None):
    """Semantic + keyword hybrid search."""
    args: dict = {"query": query, "limit": limit}
    if source_type:
        args["source_type"] = source_type
    if since:
        args["since"] = since
    _call("hybrid_search", args)


@main.command()
@click.argument("query")
@click.option("-n", "--limit", default=5, type=int, help="Max results")
@click.option("--since", default=None, help="Time filter")
def keyword(query: str, limit: int, since: str | None):
    """Full-text keyword search."""
    args: dict = {"query": query, "limit": limit}
    if since:
        args["since"] = since
    _call("keyword_search", args)


# ---------------------------------------------------------------------------
# Document commands
# ---------------------------------------------------------------------------


@main.command()
@click.argument("path")
def get(path: str):
    """Retrieve a document by source path."""
    _call("get_document", {"source_path": path})


@main.command()
@click.option("-n", "--limit", default=10, type=int, help="Max results")
def recent(limit: int):
    """Show recently indexed documents."""
    _call("recent_documents", {"limit": limit})


@main.command()
def sources():
    """Show indexed document statistics."""
    _call("list_sources")


@main.command()
def projects():
    """List known projects."""
    _call("list_projects")


@main.command()
def info():
    """Show database info and statistics."""
    _call("get_database_info")


# ---------------------------------------------------------------------------
# Write commands
# ---------------------------------------------------------------------------


@main.command()
@click.argument("title")
@click.option("--tags", default=None, help="Comma-separated tags")
@click.option("--project", default=None, help="Project name")
def save(title: str, tags: str | None, project: str | None):
    """Save knowledge from stdin.

    Example: echo 'content' | okb save 'My Title' --tags t1,t2
    """
    if sys.stdin.isatty():
        click.echo(
            "Error: Pipe content via stdin. Example: echo 'text' | okb save 'Title'",
            err=True,
        )
        sys.exit(1)
    content = sys.stdin.read().strip()
    if not content:
        click.echo("Error: No content received on stdin.", err=True)
        sys.exit(1)
    args: dict = {"title": title, "content": content}
    if tags:
        args["tags"] = [t.strip() for t in tags.split(",")]
    if project:
        args["project"] = project
    _call("save_knowledge", args)


@main.command()
@click.argument("title")
@click.option("--due", "due_date", default=None, help="Due date (YYYY-MM-DD)")
@click.option("--priority", default=None, type=int, help="Priority 1-5 (1=highest)")
@click.option("--project", default=None, help="Project name")
@click.option("--tags", default=None, help="Comma-separated tags")
def todo(title: str, due_date: str | None, priority: int | None, project: str | None,
         tags: str | None):
    """Create a TODO item."""
    args: dict = {"title": title}
    if due_date:
        args["due_date"] = due_date
    if priority:
        args["priority"] = priority
    if project:
        args["project"] = project
    if tags:
        args["tags"] = [t.strip() for t in tags.split(",")]
    _call("add_todo", args)


# ---------------------------------------------------------------------------
# Sync / rescan
# ---------------------------------------------------------------------------


@main.command()
def rescan():
    """Trigger rescan for file changes."""
    _call("trigger_rescan")


@main.group()
def sync():
    """Sync operations."""
    pass


@sync.command("run")
@click.argument("sources", nargs=-1)
@click.option("--all", "sync_all", is_flag=True, help="Sync all enabled sources")
@click.option("--full", is_flag=True, help="Full sync (ignore incremental state)")
@click.option("--repo", multiple=True, help="GitHub repo (owner/repo, can repeat)")
def sync_run(sources: tuple[str, ...], sync_all: bool, full: bool, repo: tuple[str, ...]):
    """Trigger sync for API sources."""
    args: dict = {}
    if sources:
        args["sources"] = list(sources)
    if sync_all:
        args["all"] = True
    if full:
        args["full"] = True
    if repo:
        args["repos"] = list(repo)
    _call("trigger_sync", args)


@sync.command("status")
def sync_status():
    """Show sync source status."""
    _call("list_sync_sources")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


@main.group()
def synthesize():
    """Knowledge synthesis operations."""
    pass


@synthesize.command("run")
@click.option("--project", default=None, help="Scope to project")
@click.option("--max-proposals", default=10, type=int, help="Max proposals to generate")
def synthesize_run(project: str | None, max_proposals: int):
    """Generate synthesis proposals."""
    args: dict = {"max_proposals": max_proposals}
    if project:
        args["project"] = project
    _call("synthesize_knowledge", args)


@synthesize.command("pending")
@click.option("--project", default=None, help="Filter by project")
def synthesize_pending(project: str | None):
    """List pending synthesis proposals."""
    args: dict = {}
    if project:
        args["project"] = project
    _call("list_pending_synthesis", args)


@synthesize.command("approve")
@click.argument("pending_id", type=int)
def synthesize_approve(pending_id: int):
    """Approve a synthesis proposal."""
    _call("approve_synthesis", {"pending_id": pending_id})


@synthesize.command("reject")
@click.argument("pending_id", type=int)
def synthesize_reject(pending_id: int):
    """Reject a synthesis proposal."""
    _call("reject_synthesis", {"pending_id": pending_id})


@synthesize.command("analyze")
@click.option("--project", default=None, help="Analyze specific project")
def synthesize_analyze(project: str | None):
    """Analyze knowledge base content."""
    args: dict = {}
    if project:
        args["project"] = project
    _call("analyze_knowledge_base", args)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@main.group()
def snapshot():
    """Snapshot backup/restore."""
    pass


@snapshot.command("save")
@click.argument("name", required=False)
def snapshot_save(name: str | None):
    """Create a database snapshot."""
    args: dict = {}
    if name:
        args["name"] = name
    _call("save_snapshot", args)


@snapshot.command("list")
def snapshot_list():
    """List available snapshots."""
    _call("list_snapshots")


def entry_point():
    """Entry point for the okb client CLI."""
    from .config import InsecureConfigError

    try:
        main()
    except InsecureConfigError:
        sys.exit(1)


if __name__ == "__main__":
    entry_point()
