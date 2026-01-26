"""Command-line interface for Local Knowledge Base."""

from __future__ import annotations

import importlib.resources
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click
import yaml

from .config import config, get_config_dir, get_config_path, get_default_config_yaml


@click.group()
@click.version_option(package_name="okb")
@click.option("--db", "database", default=None, help="Database to use")
@click.pass_context
def main(ctx, database):
    """Local Knowledge Base - semantic search for personal documents."""
    ctx.ensure_object(dict)
    ctx.obj["database"] = database


# =============================================================================
# Database commands
# =============================================================================


@main.group()
@click.pass_context
def db(ctx):
    """Manage the pgvector database container."""
    pass


def _check_docker() -> bool:
    """Check if docker is available."""
    return shutil.which("docker") is not None


def _get_container_status() -> str | None:
    """Get the status of the lkb container. Returns None if not found."""
    try:
        result = subprocess.run(
            [
                "docker",
                "container",
                "inspect",
                "-f",
                "{{.State.Status}}",
                config.docker_container_name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except subprocess.TimeoutExpired:
        return None


def _get_init_sql_path() -> Path:
    """Get the path to init.sql, extracting from package if needed."""
    # Try to access init.sql from package data
    try:
        ref = importlib.resources.files("okb.data").joinpath("init.sql")
        # If it's a real file path, return it directly
        with importlib.resources.as_file(ref) as path:
            return path
    except Exception:
        # Fallback: look relative to this file
        return Path(__file__).parent / "data" / "init.sql"


def _wait_for_db_ready(timeout: int = 30) -> bool:
    """Wait for database to be ready to accept connections."""
    import time

    click.echo("Waiting for database to be ready...", nl=False)
    for _ in range(timeout):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    config.docker_container_name,
                    "pg_isready",
                    "-U",
                    "knowledge",
                    "-d",
                    "knowledge_base",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                click.echo(" ready.")
                return True
        except subprocess.TimeoutExpired:
            pass
        click.echo(".", nl=False)
        time.sleep(1)
    click.echo(" timeout!")
    return False


def _run_migrations_for_db(db_cfg):
    """Run pending migrations for a specific database."""
    from .migrate import get_pending, run_migrations

    try:
        pending = get_pending(db_cfg.url)
        if pending:
            click.echo(f"  {db_cfg.name}: applying {len(pending)} migration(s)...")
            applied = run_migrations(db_cfg.url)
            for m in applied:
                click.echo(f"    ✓ {m}")
        else:
            click.echo(f"  {db_cfg.name}: up to date")
    except Exception as e:
        click.echo(f"  {db_cfg.name}: error ({e})", err=True)


def _run_migrations_all():
    """Run pending migrations on all managed databases."""
    managed_dbs = [db for db in config.databases.values() if db.managed]
    if managed_dbs:
        click.echo("Running migrations...")
        for db_cfg in managed_dbs:
            _run_migrations_for_db(db_cfg)


def _ensure_databases_exist():
    """Create databases in PostgreSQL container if they don't exist."""
    import psycopg
    from psycopg import sql

    managed_dbs = [db for db in config.databases.values() if db.managed]
    if not managed_dbs:
        return

    # Connect to postgres database (admin db) to create others
    admin_url = (
        f"postgresql://knowledge:{config.docker_password}@localhost:{config.docker_port}/postgres"
    )

    try:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            # Get existing databases
            result = conn.execute("SELECT datname FROM pg_database WHERE datistemplate = false")
            existing = {row[0] for row in result.fetchall()}

            for db_cfg in managed_dbs:
                db_name = db_cfg.database_name
                if db_name not in existing:
                    click.echo(f"Creating database: {db_name}")
                    conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))

                    # Enable pgvector extension on the new database
                    new_db_url = (
                        f"postgresql://knowledge:{config.docker_password}@"
                        f"localhost:{config.docker_port}/{db_name}"
                    )
                    with psycopg.connect(new_db_url, autocommit=True) as new_conn:
                        new_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as e:
        click.echo(f"Warning: Could not create databases: {e}", err=True)


@db.command()
def start():
    """Start the pgvector database container."""
    if not _check_docker():
        click.echo("Error: docker is not installed or not in PATH", err=True)
        sys.exit(1)

    status = _get_container_status()
    if status == "running":
        click.echo(f"Container '{config.docker_container_name}' is already running.")
        return

    if status == "exited":
        # Container exists but is stopped, start it
        click.echo(f"Starting existing container '{config.docker_container_name}'...")
        try:
            result = subprocess.run(
                ["docker", "start", config.docker_container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            click.echo("Error: docker start timed out", err=True)
            sys.exit(1)
        if result.returncode != 0:
            click.echo(f"Error starting container: {result.stderr}", err=True)
            sys.exit(1)
        click.echo("Database started.")
        _wait_for_db_ready()
        _ensure_databases_exist()
        _run_migrations_all()
        return

    # Container doesn't exist, create it
    click.echo(f"Creating container '{config.docker_container_name}'...")

    # Get init.sql path - we need to handle the case where it's in a package
    init_sql = _get_init_sql_path()

    # If init.sql is inside a zip/egg, we need to extract it to a temp location
    if not init_sql.exists():
        ref = importlib.resources.files("okb.data").joinpath("init.sql")
        init_sql_content = ref.read_text()
        # Write to temp file
        temp_dir = Path(tempfile.gettempdir()) / "okb"
        temp_dir.mkdir(exist_ok=True)
        init_sql = temp_dir / "init.sql"
        init_sql.write_text(init_sql_content)

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        config.docker_container_name,
        "-e",
        "POSTGRES_USER=knowledge",
        "-e",
        f"POSTGRES_PASSWORD={config.docker_password}",
        "-e",
        "POSTGRES_DB=knowledge_base",
        "-v",
        f"{config.docker_volume_name}:/var/lib/postgresql/data",
        "-v",
        f"{init_sql}:/docker-entrypoint-initdb.d/init.sql:ro",
        "-p",
        f"{config.docker_port}:5432",
        "--restart",
        "unless-stopped",
        "pgvector/pgvector:pg16",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        click.echo("Error: docker run timed out (may need to pull image manually)", err=True)
        sys.exit(1)
    if result.returncode != 0:
        click.echo(f"Error creating container: {result.stderr}", err=True)
        sys.exit(1)

    click.echo("Database started.")
    click.echo(f"  Container: {config.docker_container_name}")
    click.echo(f"  Port: {config.docker_port}")
    click.echo(f"  Volume: {config.docker_volume_name}")
    _wait_for_db_ready()
    _ensure_databases_exist()
    _run_migrations_all()


@db.command()
def stop():
    """Stop the pgvector database container."""
    if not _check_docker():
        click.echo("Error: docker is not installed or not in PATH", err=True)
        sys.exit(1)

    status = _get_container_status()
    if status is None:
        click.echo(f"Container '{config.docker_container_name}' does not exist.")
        return

    if status != "running":
        click.echo(f"Container '{config.docker_container_name}' is not running (status: {status}).")
        return

    click.echo(f"Stopping container '{config.docker_container_name}'...")
    try:
        result = subprocess.run(
            ["docker", "stop", config.docker_container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        click.echo("Error: docker stop timed out", err=True)
        sys.exit(1)
    if result.returncode != 0:
        click.echo(f"Error stopping container: {result.stderr}", err=True)
        sys.exit(1)

    click.echo("Database stopped.")


@db.command()
def status():
    """Show database container status."""
    if not _check_docker():
        click.echo("Error: docker is not installed or not in PATH", err=True)
        sys.exit(1)

    container_status = _get_container_status()
    if container_status is None:
        click.echo(f"Container '{config.docker_container_name}' does not exist.")
        click.echo("Run 'okb db start' to create it.")
        return

    click.echo(f"Container: {config.docker_container_name}")
    click.echo(f"Status: {container_status}")
    click.echo(f"Port: {config.docker_port}")
    click.echo(f"Volume: {config.docker_volume_name}")

    if container_status == "running":
        # Try to get more info
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    config.docker_container_name,
                    "pg_isready",
                    "-U",
                    "knowledge",
                    "-d",
                    "knowledge_base",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            click.echo("Database: check timed out")
            return
        if result.returncode == 0:
            click.echo("Database: ready")
            # Show migration status
            try:
                from .migrate import get_applied, get_pending

                applied = get_applied(config.db_url)
                pending = get_pending(config.db_url)
                click.echo(f"Migrations: {len(applied)} applied, {len(pending)} pending")
                if pending:
                    click.echo("  Run 'okb db migrate' to apply pending migrations.")
            except Exception as e:
                click.echo(f"Migrations: error checking ({e})")
        else:
            click.echo("Database: not ready")


@db.command()
@click.argument("name", required=False)
def migrate(name):
    """Apply pending database migrations.

    If NAME is provided, migrate only that database.
    Otherwise, migrate all configured databases.

    Creates missing databases automatically for managed databases.
    """
    # Ensure managed databases exist before migrating
    _ensure_databases_exist()

    if name:
        # Migrate specific database
        try:
            db_cfg = config.get_database(name)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        _run_migrations_for_db(db_cfg)
    else:
        # Migrate all databases
        for db_cfg in config.databases.values():
            _run_migrations_for_db(db_cfg)
    click.echo("Done.")


@db.command()
def destroy():
    """Remove the database container and volume (destructive!)."""
    if not _check_docker():
        click.echo("Error: docker is not installed or not in PATH", err=True)
        sys.exit(1)

    if not click.confirm(
        f"This will delete container '{config.docker_container_name}' and volume "
        f"'{config.docker_volume_name}'. All data will be lost. Continue?"
    ):
        return

    # Stop and remove container
    subprocess.run(
        ["docker", "rm", "-f", config.docker_container_name],
        capture_output=True,
        timeout=30,
    )
    click.echo(f"Removed container '{config.docker_container_name}'.")

    # Remove volume
    subprocess.run(
        ["docker", "volume", "rm", config.docker_volume_name],
        capture_output=True,
        timeout=30,
    )
    click.echo(f"Removed volume '{config.docker_volume_name}'.")


@db.command("list")
def db_list():
    """List all configured databases."""
    click.echo("Configured databases:")
    for name, db_cfg in config.databases.items():
        markers = []
        if db_cfg.default:
            markers.append("default")
        markers.append("managed" if db_cfg.managed else "external")
        click.echo(f"  {name} [{', '.join(markers)}]")
        click.echo(f"    URL: {db_cfg.url}")


# =============================================================================
# Config commands
# =============================================================================


@main.group("config")
def config_cmd():
    """Manage configuration."""
    pass


@config_cmd.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing config file")
def config_init(force: bool):
    """Create default config file at ~/.config/okb/config.yaml."""
    config_path = get_config_path()

    if config_path.exists() and not force:
        click.echo(f"Config file already exists at {config_path}")
        click.echo("Use --force to overwrite.")
        return

    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path.write_text(get_default_config_yaml())
    click.echo(f"Created config file at {config_path}")


@config_cmd.command("show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def config_show(as_json: bool):
    """Show current configuration."""
    config_path = get_config_path()

    if as_json:
        click.echo(json.dumps(config.to_dict(), indent=2))
    else:
        click.echo(f"Config file: {config_path}")
        click.echo(f"  Exists: {config_path.exists()}")
        click.echo("")
        click.echo(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))


@config_cmd.command("path")
def config_path_cmd():
    """Print the config file path."""
    click.echo(get_config_path())


# =============================================================================
# Ingest command
# =============================================================================


@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--metadata", "-m", default="{}", help="JSON metadata to attach")
@click.option("--local", is_flag=True, help="Use local CPU embedding instead of Modal")
@click.option("--db", "database", default=None, help="Database to ingest into")
@click.pass_context
def ingest(ctx, paths: tuple[str, ...], metadata: str, local: bool, database: str | None):
    """Ingest documents or URLs into the knowledge base."""
    import json as json_module
    from pathlib import Path

    from .ingest import (
        Ingester,
        check_file_skip,
        collect_documents,
        is_text_file,
        is_url,
        parse_document,
        parse_url,
    )

    try:
        extra_metadata = json_module.loads(metadata)
    except json_module.JSONDecodeError as e:
        click.echo(f"Error parsing metadata JSON: {e}", err=True)
        sys.exit(1)

    # Get database URL from --db option or context
    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)
    ingester = Ingester(db_cfg.url, use_modal=not local)

    documents = []
    for path_str in paths:
        # Check if it's a URL first
        if is_url(path_str):
            click.echo(f"Fetching: {path_str}")
            try:
                documents.append(parse_url(path_str, extra_metadata))
            except Exception as e:
                click.echo(f"Error fetching URL: {e}", err=True)
            continue

        path = Path(path_str).resolve()
        if path.is_dir():
            documents.extend(collect_documents(path, extra_metadata))
        elif path.is_file():
            # Check security patterns first
            skip_check = check_file_skip(path)
            if skip_check.should_skip:
                prefix = "BLOCKED" if skip_check.is_security else "Skipping"
                click.echo(f"{prefix}: {path} ({skip_check.reason})", err=True)
                continue

            # For explicitly provided files, try to parse even with unknown extension
            # Always allow .pdf and .docx even if not in config (user may have old config)
            if path.suffix in config.all_extensions or path.suffix in (".pdf", ".docx"):
                try:
                    documents.extend(parse_document(path, extra_metadata))
                except ValueError as e:
                    click.echo(f"Skipping: {e}", err=True)
                    continue
            elif is_text_file(path):
                # Unknown extension but appears to be text - parse as code/config
                click.echo(f"Parsing as text: {path}")
                documents.extend(parse_document(path, extra_metadata, force=True))
            else:
                click.echo(f"Skipping binary file: {path}", err=True)
        else:
            click.echo(f"Not found: {path_str}", err=True)

    if not documents:
        click.echo("No documents found to ingest.")
        return

    click.echo(f"Found {len(documents)} documents to process")
    ingester.ingest_documents(documents)
    click.echo("Done!")


# =============================================================================
# Rescan command
# =============================================================================


@main.command()
@click.option("--db", "database", default=None, help="Database to rescan")
@click.option("--local", is_flag=True, help="Use local CPU embedding instead of Modal")
@click.option("--dry-run", is_flag=True, help="Show changes without executing")
@click.option("--delete", "delete_missing", is_flag=True, help="Remove documents for missing files")
@click.pass_context
def rescan(ctx, database: str | None, local: bool, dry_run: bool, delete_missing: bool):
    """Check indexed documents for freshness and re-ingest changed ones.

    Compares stored file modification times against actual file mtimes.
    Files that have changed are deleted and re-ingested. Missing files
    are reported (use --delete to remove them from the index).

    Examples:

        okb rescan              # Rescan default database

        okb rescan --dry-run    # Show what would change

        okb rescan --delete     # Also remove missing files

        okb rescan --db work    # Rescan specific database
    """
    from .rescan import Rescanner

    # Get database URL from --db option or context
    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    click.echo(f"Scanning database '{db_cfg.name}'...")
    if dry_run:
        click.echo("(dry run - no changes will be made)")

    rescanner = Rescanner(db_cfg.url, use_modal=not local)
    result = rescanner.rescan(dry_run=dry_run, delete_missing=delete_missing, verbose=True)

    # Print summary
    click.echo("")
    summary_parts = []
    if result.updated:
        summary_parts.append(f"{len(result.updated)} updated")
    if result.deleted:
        summary_parts.append(f"{len(result.deleted)} deleted")
    if result.missing:
        summary_parts.append(f"{len(result.missing)} missing")
    summary_parts.append(f"{result.unchanged} unchanged")

    if result.errors:
        summary_parts.append(f"{len(result.errors)} errors")

    click.echo(f"Summary: {', '.join(summary_parts)}")

    if result.missing and not delete_missing:
        click.echo("Use --delete to remove missing files from the index.")


# =============================================================================
# Serve command
# =============================================================================


@main.command()
@click.option("--db", "database", default=None, help="Database to serve")
@click.option("--http", "use_http", is_flag=True, help="Use HTTP transport instead of stdio")
@click.option("--host", default=None, help="HTTP server host (default: 127.0.0.1)")
@click.option("--port", type=int, default=None, help="HTTP server port (default: 8080)")
@click.pass_context
def serve(ctx, database: str | None, use_http: bool, host: str | None, port: int | None):
    """Start the MCP server for Claude Code integration.

    By default, runs in stdio mode for direct Claude Code integration.
    Use --http to run as an HTTP server with token authentication.
    """
    import asyncio

    if use_http:
        from .http_server import run_http_server

        http_host = host or config.http_host
        http_port = port or config.http_port
        run_http_server(host=http_host, port=http_port)
    else:
        from .mcp_server import main as mcp_main

        # Get database URL from --db option or context
        db_name = database or ctx.obj.get("database")
        db_cfg = config.get_database(db_name)
        asyncio.run(mcp_main(db_cfg.url))


# =============================================================================
# Watch command
# =============================================================================


@main.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--metadata", "-m", default="{}", help="JSON metadata to attach")
@click.option("--local", is_flag=True, help="Use local CPU embedding instead of Modal")
@click.option("--db", "database", default=None, help="Database to watch for")
@click.pass_context
def watch(ctx, paths: tuple[str, ...], metadata: str, local: bool, database: str | None):
    """Watch directories for changes and auto-ingest."""
    from .scripts.watch import main as watch_main

    # Get database URL from --db option or context
    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    # Convert to the format watch.py expects
    sys.argv = ["okb-watch"] + list(paths)
    sys.argv.extend(["--db-url", db_cfg.url])
    if metadata != "{}":
        sys.argv.extend(["--metadata", metadata])
    if local:
        sys.argv.append("--local")

    watch_main()


# =============================================================================
# Modal commands
# =============================================================================


@main.group()
def modal():
    """Manage Modal GPU embedder."""
    pass


@modal.command()
def deploy():
    """Deploy embedder to Modal."""
    if not shutil.which("modal"):
        click.echo("Error: modal CLI is not installed.", err=True)
        click.echo("Install with: pip install modal", err=True)
        sys.exit(1)

    # Find modal_embedder.py in the package
    embedder_path = Path(__file__).parent / "modal_embedder.py"
    if not embedder_path.exists():
        click.echo(f"Error: modal_embedder.py not found at {embedder_path}", err=True)
        sys.exit(1)

    click.echo(f"Deploying {embedder_path} to Modal...")
    result = subprocess.run(
        ["modal", "deploy", str(embedder_path)],
        cwd=embedder_path.parent,
    )
    sys.exit(result.returncode)


# =============================================================================
# Sync commands (plugin system)
# =============================================================================


@main.group()
def sync():
    """Sync data from external API sources (plugins)."""
    pass


def _get_sync_state(conn, source_name: str, db_name: str):
    """Get sync state from database."""
    from .plugins.base import SyncState

    result = conn.execute(
        """SELECT last_sync, cursor, extra FROM sync_state
           WHERE source_name = %s AND database_name = %s""",
        (source_name, db_name),
    ).fetchone()

    if result:
        return SyncState(
            last_sync=result["last_sync"],
            cursor=result["cursor"],
            extra=result["extra"] or {},
        )
    return None


def _save_sync_state(conn, source_name: str, db_name: str, state):
    """Save sync state to database."""
    conn.execute(
        """INSERT INTO sync_state (source_name, database_name, last_sync, cursor, extra, updated_at)
           VALUES (%s, %s, %s, %s, %s, NOW())
           ON CONFLICT (source_name, database_name)
           DO UPDATE SET last_sync = EXCLUDED.last_sync,
                        cursor = EXCLUDED.cursor,
                        extra = EXCLUDED.extra,
                        updated_at = NOW()""",
        (source_name, db_name, state.last_sync, state.cursor, json.dumps(state.extra)),
    )
    conn.commit()


def _apply_llm_filter(documents: list, filter_cfg: dict, source_name: str) -> list:
    """Apply LLM filtering to documents.

    Args:
        documents: List of documents to filter
        filter_cfg: Filter configuration with 'prompt' and 'action_on_skip'
        source_name: Name of the source (for logging)

    Returns:
        Filtered list of documents
    """
    from .llm import FilterAction, filter_document

    custom_prompt = filter_cfg.get("prompt")
    action_on_skip = filter_cfg.get("action_on_skip", "discard")

    filtered = []
    skipped = 0
    review = 0

    for doc in documents:
        result = filter_document(doc, custom_prompt=custom_prompt)

        if result.action == FilterAction.SKIP:
            skipped += 1
            if action_on_skip == "archive":
                # Store without embedding (future: add flag to document)
                pass
            # Otherwise discard
            continue
        elif result.action == FilterAction.REVIEW:
            review += 1
            # Still ingest, but could flag for review (future: add metadata)

        filtered.append(doc)

    if skipped or review:
        click.echo(f"  Filter: {len(filtered)} ingested, {skipped} skipped, {review} for review")

    return filtered


@sync.command("run")
@click.argument("sources", nargs=-1)
@click.option("--all", "sync_all", is_flag=True, help="Sync all enabled sources")
@click.option("--full", is_flag=True, help="Ignore incremental state, do full sync")
@click.option("--local", is_flag=True, help="Use local CPU embedding instead of Modal")
@click.option("--db", "database", default=None, help="Database to sync into")
@click.option("--folder", multiple=True, help="Filter to specific folder path (can repeat)")
@click.option("--doc", "doc_ids", multiple=True, help="Sync specific document ID (can repeat)")
# GitHub-specific options
@click.option("--repo", multiple=True, help="GitHub repo to sync (owner/repo, can repeat)")
@click.option(
    "--source", "include_source", is_flag=True, help="Sync all source files (not just README+docs)"
)
@click.option("--issues", "include_issues", is_flag=True, help="Include GitHub issues")
@click.option("--prs", "include_prs", is_flag=True, help="Include GitHub pull requests")
@click.option("--wiki", "include_wiki", is_flag=True, help="Include GitHub wiki pages")
@click.pass_context
def sync_run(
    ctx,
    sources: tuple[str, ...],
    sync_all: bool,
    full: bool,
    local: bool,
    database: str | None,
    folder: tuple[str, ...],
    doc_ids: tuple[str, ...],
    repo: tuple[str, ...],
    include_source: bool,
    include_issues: bool,
    include_prs: bool,
    include_wiki: bool,
):
    """Sync from API sources.

    Example: lkb sync run github --repo owner/repo
    """
    import psycopg
    from psycopg.rows import dict_row

    from .ingest import Ingester
    from .plugins.registry import PluginRegistry

    # Get database
    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    # Determine which sources to sync
    if sync_all:
        source_names = config.list_enabled_sources()
    elif sources:
        source_names = list(sources)
    else:
        click.echo("Error: Specify sources to sync or use --all", err=True)
        click.echo("Available sources: ", nl=False)
        click.echo(", ".join(PluginRegistry.list_sources()) or "(none installed)")
        sys.exit(1)

    if not source_names:
        click.echo("No sources to sync.")
        return

    ingester = Ingester(db_cfg.url, use_modal=not local)

    with psycopg.connect(db_cfg.url, row_factory=dict_row) as conn:
        for source_name in source_names:
            # Get the plugin
            source = PluginRegistry.get_source(source_name)
            if source is None:
                click.echo(f"Error: Source '{source_name}' not found.", err=True)
                click.echo(f"Installed sources: {', '.join(PluginRegistry.list_sources())}")
                continue

            # Get and resolve config
            source_cfg = config.get_source_config(source_name)
            if source_cfg is None:
                click.echo(f"Skipping '{source_name}': not configured or disabled", err=True)
                continue

            # Merge CLI options into config (override config file values)
            if folder:
                source_cfg["folders"] = list(folder)
            if doc_ids:
                source_cfg["doc_ids"] = list(doc_ids)
            # GitHub-specific options
            if repo:
                source_cfg["repos"] = list(repo)
            if include_source:
                source_cfg["include_source"] = True
            if include_issues:
                source_cfg["include_issues"] = True
            if include_prs:
                source_cfg["include_prs"] = True
            if include_wiki:
                source_cfg["include_wiki"] = True

            try:
                source.configure(source_cfg)
            except Exception as e:
                click.echo(f"Error configuring '{source_name}': {e}", err=True)
                continue

            # Get sync state (unless --full)
            state = None if full else _get_sync_state(conn, source_name, db_cfg.name)

            click.echo(f"Syncing {source_name}..." + (" (full)" if full else ""))

            try:
                documents, new_state = source.fetch(state)
            except Exception as e:
                click.echo(f"Error fetching from '{source_name}': {e}", err=True)
                continue

            if documents:
                click.echo(f"  Fetched {len(documents)} documents")

                # Apply LLM filtering if configured
                llm_filter_cfg = source_cfg.get("llm_filter", {})
                if llm_filter_cfg.get("enabled"):
                    documents = _apply_llm_filter(
                        documents,
                        llm_filter_cfg,
                        source_name,
                    )

                if documents:
                    ingester.ingest_documents(documents)
                else:
                    click.echo("  All documents filtered out")
            else:
                click.echo("  No new documents")

            # Save state
            _save_sync_state(conn, source_name, db_cfg.name, new_state)

    click.echo("Done!")


@sync.command("list")
def sync_list():
    """List available API sources."""
    from .plugins.registry import PluginRegistry

    installed = PluginRegistry.list_sources()
    configured = config.list_enabled_sources()

    click.echo("Installed sources:")
    if installed:
        for name in installed:
            status = "configured" if name in configured else "not configured"
            click.echo(f"  {name} [{status}]")
    else:
        click.echo("  (none)")

    # Show configured but not installed
    not_installed = set(configured) - set(installed)
    if not_installed:
        click.echo("\nConfigured but not installed:")
        for name in not_installed:
            click.echo(f"  {name}")


@sync.command("list-projects")
@click.argument("source")
def sync_list_projects(source: str):
    """List projects from an API source (for finding project IDs).

    Example: okb sync list-projects todoist
    """
    from .plugins.registry import PluginRegistry

    # Get the plugin
    source_obj = PluginRegistry.get_source(source)
    if source_obj is None:
        click.echo(f"Error: Source '{source}' not found.", err=True)
        click.echo(f"Installed sources: {', '.join(PluginRegistry.list_sources())}")
        sys.exit(1)

    # Check if source supports list_projects
    if not hasattr(source_obj, "list_projects"):
        click.echo(f"Error: Source '{source}' does not support listing projects.", err=True)
        sys.exit(1)

    # Get and resolve config
    source_cfg = config.get_source_config(source)
    if source_cfg is None:
        click.echo(f"Error: Source '{source}' not configured.", err=True)
        click.echo("Add it to your config file under plugins.sources")
        sys.exit(1)

    try:
        source_obj.configure(source_cfg)
    except Exception as e:
        click.echo(f"Error configuring '{source}': {e}", err=True)
        sys.exit(1)

    try:
        projects = source_obj.list_projects()
        if projects:
            click.echo(f"Projects in {source}:")
            for project_id, name in projects:
                click.echo(f"  {project_id}: {name}")
        else:
            click.echo("No projects found.")
    except Exception as e:
        click.echo(f"Error listing projects: {e}", err=True)
        sys.exit(1)


@sync.command("auth")
@click.argument("source")
def sync_auth(source: str):
    """Authenticate with an API source (get tokens).

    Currently supports: dropbox-paper

    Example: okb sync auth dropbox-paper
    """
    if source == "dropbox-paper":
        _auth_dropbox()
    else:
        click.echo(f"Error: Authentication helper not available for '{source}'", err=True)
        click.echo("Supported: dropbox-paper")
        sys.exit(1)


def _auth_dropbox():
    """Interactive OAuth flow for Dropbox."""
    try:
        import dropbox
        from dropbox import DropboxOAuth2FlowNoRedirect
    except ImportError:
        click.echo("Error: dropbox package not installed", err=True)
        click.echo("Install with: pip install dropbox", err=True)
        sys.exit(1)

    click.echo("Dropbox OAuth Setup")
    click.echo("=" * 50)
    click.echo("")
    click.echo("You'll need your Dropbox app credentials.")
    click.echo("Get them at: https://www.dropbox.com/developers/apps")
    click.echo("")

    app_key = click.prompt("App key")
    app_secret = click.prompt("App secret")

    # Start OAuth flow
    auth_flow = DropboxOAuth2FlowNoRedirect(
        app_key,
        app_secret,
        token_access_type="offline",  # This gives us a refresh token
    )

    authorize_url = auth_flow.start()
    click.echo("")
    click.echo("1. Go to this URL in your browser:")
    click.echo(f"   {authorize_url}")
    click.echo("")
    click.echo("2. Click 'Allow' to authorize the app")
    click.echo("3. Copy the authorization code")
    click.echo("")

    auth_code = click.prompt("Enter the authorization code")

    try:
        oauth_result = auth_flow.finish(auth_code.strip())
    except Exception as e:
        click.echo(f"Error: Failed to get tokens - {e}", err=True)
        sys.exit(1)

    click.echo("")
    click.echo("Success! Add these to your environment or config:")
    click.echo("")
    click.echo(f"DROPBOX_APP_KEY={app_key}")
    click.echo(f"DROPBOX_APP_SECRET={app_secret}")
    click.echo(f"DROPBOX_REFRESH_TOKEN={oauth_result.refresh_token}")
    click.echo("")
    click.echo("Config example (~/.config/okb/config.yaml):")
    click.echo("")
    click.echo("plugins:")
    click.echo("  sources:")
    click.echo("    dropbox-paper:")
    click.echo("      enabled: true")
    click.echo("      app_key: ${DROPBOX_APP_KEY}")
    click.echo("      app_secret: ${DROPBOX_APP_SECRET}")
    click.echo("      refresh_token: ${DROPBOX_REFRESH_TOKEN}")


@sync.command("status")
@click.argument("source", required=False)
@click.option("--db", "database", default=None, help="Database to check")
@click.pass_context
def sync_status(ctx, source: str | None, database: str | None):
    """Show sync status and last sync times."""
    import psycopg
    from psycopg.rows import dict_row

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    with psycopg.connect(db_cfg.url, row_factory=dict_row) as conn:
        if source:
            # Show status for specific source
            result = conn.execute(
                """SELECT source_name, last_sync, cursor, extra, updated_at
                   FROM sync_state
                   WHERE source_name = %s AND database_name = %s""",
                (source, db_cfg.name),
            ).fetchone()

            if result:
                click.echo(f"Source: {result['source_name']}")
                click.echo(f"  Last sync: {result['last_sync'] or 'never'}")
                click.echo(f"  Updated: {result['updated_at']}")
                if result["cursor"]:
                    click.echo(f"  Cursor: {result['cursor'][:50]}...")
            else:
                click.echo(f"No sync history for '{source}'")

            # Show document count
            doc_count = conn.execute(
                """SELECT COUNT(*) as count FROM documents
                   WHERE metadata->>'sync_source' = %s""",
                (source,),
            ).fetchone()
            click.echo(f"  Documents: {doc_count['count']}")
        else:
            # Show all sync states
            results = conn.execute(
                """SELECT source_name, last_sync, updated_at
                   FROM sync_state
                   WHERE database_name = %s
                   ORDER BY updated_at DESC""",
                (db_cfg.name,),
            ).fetchall()

            if results:
                click.echo(f"Sync status for database '{db_cfg.name}':")
                for row in results:
                    if row["last_sync"]:
                        last = row["last_sync"].strftime("%Y-%m-%d %H:%M")
                    else:
                        last = "never"
                    click.echo(f"  {row['source_name']}: {last}")
            else:
                click.echo("No sync history")


# =============================================================================
# Token commands
# =============================================================================


@main.group()
def token():
    """Manage API tokens for HTTP access."""
    pass


@token.command("create")
@click.option("--db", "database", default=None, help="Database to create token for")
@click.option("--ro", "read_only", is_flag=True, help="Create read-only token (default: rw)")
@click.option("-d", "--description", default=None, help="Token description")
@click.pass_context
def token_create(ctx, database: str | None, read_only: bool, description: str | None):
    """Create a new API token."""
    from .tokens import create_token

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)
    permissions = "ro" if read_only else "rw"

    try:
        token = create_token(db_cfg.url, db_cfg.name, permissions, description)
        click.echo(f"Token created for database '{db_cfg.name}' ({permissions}):")
        click.echo(f"  {token}")
        click.echo("")
        click.echo("Save this token - it cannot be retrieved later.")
    except Exception as e:
        click.echo(f"Error creating token: {e}", err=True)
        sys.exit(1)


@token.command("list")
@click.option("--db", "database", default=None, help="Database to list tokens for")
@click.pass_context
def token_list(ctx, database: str | None):
    """List all tokens for a database."""
    from .tokens import list_tokens

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    try:
        tokens = list_tokens(db_cfg.url)
        if not tokens:
            click.echo(f"No tokens found for database '{db_cfg.name}'")
            return

        click.echo(f"Tokens for database '{db_cfg.name}':")
        for t in tokens:
            desc = f" - {t.description}" if t.description else ""
            last_used = t.last_used_at.strftime("%Y-%m-%d %H:%M") if t.last_used_at else "never"
            click.echo(f"  [{t.permissions}] {t.token_hash[:12]}...{desc}")
            created = t.created_at.strftime("%Y-%m-%d %H:%M")
            click.echo(f"      Created: {created}, Last used: {last_used}")
    except Exception as e:
        click.echo(f"Error listing tokens: {e}", err=True)
        sys.exit(1)


@token.command("revoke")
@click.argument("token_value")
@click.option("--db", "database", default=None, help="Database to revoke token from")
@click.pass_context
def token_revoke(ctx, token_value: str, database: str | None):
    """Revoke (delete) an API token.

    TOKEN_VALUE must be the full token string.
    """
    from .tokens import delete_token

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    try:
        deleted = delete_token(db_cfg.url, token_value)
        if deleted:
            click.echo("Token revoked.")
        else:
            click.echo("Token not found. Make sure you're using the full token string.", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error revoking token: {e}", err=True)
        sys.exit(1)


# =============================================================================
# LLM commands
# =============================================================================


@main.group()
def llm():
    """Manage LLM integration for document classification."""
    pass


@llm.command("status")
@click.option("--db", "database", default=None, help="Database to check cache for")
@click.pass_context
def llm_status(ctx, database: str | None):
    """Show LLM configuration and connectivity status.

    Displays current provider settings, tests connectivity,
    and shows cache statistics.
    """
    import os

    click.echo("LLM Configuration")
    click.echo("-" * 40)

    # Show config
    click.echo(f"Provider: {config.llm_provider or '(disabled)'}")
    if config.llm_provider:
        click.echo(f"Model: {config.llm_model}")
        click.echo(f"Timeout: {config.llm_timeout}s")
        click.echo(f"Cache responses: {config.llm_cache_responses}")

        if config.llm_provider == "modal":
            click.echo("Backend: Modal GPU (deploy with: lkb llm deploy)")
        elif config.llm_use_bedrock:
            click.echo(f"Backend: AWS Bedrock (region: {config.llm_aws_region})")
        else:
            api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
            click.echo(f"API key set: {'yes' if api_key_set else 'no (set ANTHROPIC_API_KEY)'}")

    click.echo("")

    # Test connectivity if provider is configured
    if config.llm_provider:
        click.echo("Connectivity Test")
        click.echo("-" * 40)
        try:
            from .llm.providers import get_provider

            provider = get_provider()
            if provider is None:
                click.echo("Status: provider initialization failed")
            elif provider.is_available():
                click.echo("Status: available")
                # List models
                if hasattr(provider, "list_models"):
                    models = provider.list_models()
                    click.echo(f"Available models: {', '.join(models[:3])}...")
            else:
                click.echo("Status: not available (check API key or credentials)")
        except ImportError:
            click.echo("Status: missing dependencies")
            click.echo("  Install with: pip install 'okb[llm]'")
        except Exception as e:
            click.echo(f"Status: error - {e}")

    # Show cache stats if database is available
    click.echo("")
    click.echo("Cache Statistics")
    click.echo("-" * 40)
    try:
        db_name = database or ctx.obj.get("database")
        db_cfg = config.get_database(db_name)

        from .llm.cache import get_cache_stats

        stats = get_cache_stats(db_cfg.url)
        click.echo(f"Total cached responses: {stats['total_entries']}")
        if stats["by_provider"]:
            for entry in stats["by_provider"]:
                click.echo(f"  {entry['provider']}/{entry['model']}: {entry['count']}")
        if stats["oldest_entry"]:
            click.echo(f"Oldest entry: {stats['oldest_entry']}")
    except Exception as e:
        click.echo(f"Cache unavailable: {e}")


@llm.command("clear-cache")
@click.option("--db", "database", default=None, help="Database to clear cache for")
@click.option(
    "--older-than", "days", type=int, default=None, help="Only clear entries older than N days"
)
@click.option("--yes", is_flag=True, help="Skip confirmation")
@click.pass_context
def llm_clear_cache(ctx, database: str | None, days: int | None, yes: bool):
    """Clear the LLM response cache."""
    from datetime import UTC, datetime, timedelta

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    if days:
        older_than = datetime.now(UTC) - timedelta(days=days)
        msg = f"Clear cache entries older than {days} days?"
    else:
        older_than = None
        msg = "Clear ALL cache entries?"

    if not yes:
        if not click.confirm(msg):
            click.echo("Cancelled.")
            return

    from .llm.cache import clear_cache

    deleted = clear_cache(older_than=older_than, db_url=db_cfg.url)
    click.echo(f"Deleted {deleted} cache entries.")


@llm.command("deploy")
def llm_deploy():
    """Deploy the Modal LLM app for open model inference.

    This deploys a GPU-accelerated LLM service on Modal using Llama 3.2.
    Required for using provider: modal in your config.

    Requires Modal CLI to be installed and authenticated:
        pip install modal
        modal token new
    """
    if not shutil.which("modal"):
        click.echo("Error: modal CLI is not installed.", err=True)
        click.echo("Install with: pip install modal", err=True)
        click.echo("Then authenticate: modal token new", err=True)
        sys.exit(1)

    # Find modal_llm.py in the package
    llm_path = Path(__file__).parent / "modal_llm.py"
    if not llm_path.exists():
        click.echo(f"Error: modal_llm.py not found at {llm_path}", err=True)
        sys.exit(1)

    click.echo(f"Deploying {llm_path} to Modal...")
    click.echo("Note: First deploy downloads the model (~2GB) and may take a few minutes.")
    result = subprocess.run(
        ["modal", "deploy", str(llm_path)],
        cwd=llm_path.parent,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
