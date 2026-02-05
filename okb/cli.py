"""Command-line interface for Local Knowledge Base."""

from __future__ import annotations

import importlib.resources
import json
import os
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
    """Get the status of the okb container. Returns None if not found."""
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
# Snapshot commands
# =============================================================================


@db.group()
@click.pass_context
def snapshot(ctx):
    """Manage database snapshots."""
    pass


def _get_snapshot_path(db_cfg, name: str) -> Path:
    """Get the full path for a snapshot file."""
    from okb.config import get_snapshots_dir

    snapshots_dir = get_snapshots_dir(db_cfg.database_name)
    return snapshots_dir / f"{name}.dump"


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@snapshot.command("save")
@click.argument("name", required=False)
@click.pass_context
def snapshot_save(ctx, name):
    """Create a database snapshot.

    NAME is optional; defaults to timestamp (e.g., 20250204T143022).
    """
    if not _check_docker():
        click.echo("Error: docker is not installed or not in PATH", err=True)
        sys.exit(1)

    status = _get_container_status()
    if status != "running":
        click.echo("Error: database container is not running", err=True)
        sys.exit(1)

    db_name = ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    if not db_cfg.managed:
        click.echo(f"Error: database '{db_cfg.name}' is not managed by okb", err=True)
        sys.exit(1)

    # Generate name if not provided
    if not name:
        from datetime import datetime

        name = datetime.now().strftime("%Y%m%dT%H%M%S")

    snapshot_path = _get_snapshot_path(db_cfg, name)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    if snapshot_path.exists():
        click.echo(f"Error: snapshot '{name}' already exists", err=True)
        sys.exit(1)

    click.echo(f"Creating snapshot '{name}' for database '{db_cfg.database_name}'...")

    # Run pg_dump inside container
    result = subprocess.run(
        [
            "docker",
            "exec",
            config.docker_container_name,
            "pg_dump",
            "-U",
            "knowledge",
            "-Fc",  # Custom format (compressed, supports pg_restore)
            db_cfg.database_name,
        ],
        capture_output=True,
        timeout=600,  # 10 minute timeout for large databases
    )

    if result.returncode != 0:
        click.echo(f"Error: pg_dump failed: {result.stderr.decode()}", err=True)
        sys.exit(1)

    # Write to file
    snapshot_path.write_bytes(result.stdout)
    size = _format_size(snapshot_path.stat().st_size)
    click.echo(f"Saved snapshot: {snapshot_path} ({size})")


@snapshot.command("list")
@click.pass_context
def snapshot_list(ctx):
    """List available snapshots."""
    from okb.config import get_snapshots_dir

    db_name = ctx.obj.get("database")
    db_cfg = config.get_database(db_name)
    snapshots_dir = get_snapshots_dir(db_cfg.database_name)

    if not snapshots_dir.exists():
        click.echo(f"No snapshots for database '{db_cfg.database_name}'")
        return

    snapshots = sorted(snapshots_dir.glob("*.dump"))
    if not snapshots:
        click.echo(f"No snapshots for database '{db_cfg.database_name}'")
        return

    click.echo(f"Snapshots for '{db_cfg.database_name}':")
    for snap in snapshots:
        stat = snap.stat()
        size = _format_size(stat.st_size)
        from datetime import datetime

        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        name = snap.stem
        click.echo(f"  {name}  {size:>10}  {mtime}")


@snapshot.command("restore")
@click.argument("name")
@click.option("--no-backup", is_flag=True, help="Skip creating pre-restore backup")
@click.pass_context
def snapshot_restore(ctx, name, no_backup):
    """Restore database from a snapshot.

    WARNING: This will replace all data in the database.

    By default, creates a pre-restore backup before restoring.
    Use --no-backup to skip the backup step.
    """
    if not _check_docker():
        click.echo("Error: docker is not installed or not in PATH", err=True)
        sys.exit(1)

    status = _get_container_status()
    if status != "running":
        click.echo("Error: database container is not running", err=True)
        sys.exit(1)

    db_name = ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    if not db_cfg.managed:
        click.echo(f"Error: database '{db_cfg.name}' is not managed by okb", err=True)
        sys.exit(1)

    snapshot_path = _get_snapshot_path(db_cfg, name)
    if not snapshot_path.exists():
        click.echo(f"Error: snapshot '{name}' not found", err=True)
        sys.exit(1)

    if not click.confirm(
        f"This will replace ALL data in database '{db_cfg.database_name}' with snapshot '{name}'. "
        "Continue?"
    ):
        return

    # Create pre-restore backup unless --no-backup is set
    if not no_backup:
        from datetime import datetime

        backup_name = f"pre-restore-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        backup_path = _get_snapshot_path(db_cfg, backup_name)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        click.echo(f"Creating pre-restore backup '{backup_name}'...")
        backup_result = subprocess.run(
            [
                "docker",
                "exec",
                config.docker_container_name,
                "pg_dump",
                "-U",
                "knowledge",
                "-Fc",
                db_cfg.database_name,
            ],
            capture_output=True,
            timeout=600,
        )

        if backup_result.returncode != 0:
            click.echo(
                f"Warning: pre-restore backup failed: {backup_result.stderr.decode()}", err=True
            )
        else:
            backup_path.write_bytes(backup_result.stdout)
            size = _format_size(backup_path.stat().st_size)
            click.echo(f"Created pre-restore backup: {backup_name} ({size})")

    click.echo(f"Restoring '{name}' to database '{db_cfg.database_name}'...")

    # Read snapshot and pipe to pg_restore
    snapshot_data = snapshot_path.read_bytes()

    result = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            config.docker_container_name,
            "pg_restore",
            "-U",
            "knowledge",
            "-d",
            db_cfg.database_name,
            "--clean",
            "--if-exists",
        ],
        input=snapshot_data,
        capture_output=True,
        timeout=600,
    )

    # pg_restore may return warnings even on success
    if result.returncode != 0 and b"error" in result.stderr.lower():
        click.echo(f"Error: pg_restore failed: {result.stderr.decode()}", err=True)
        sys.exit(1)

    click.echo("Restore complete.")


@snapshot.command("delete")
@click.argument("name")
@click.pass_context
def snapshot_delete(ctx, name):
    """Delete a snapshot."""
    db_name = ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    snapshot_path = _get_snapshot_path(db_cfg, name)
    if not snapshot_path.exists():
        click.echo(f"Error: snapshot '{name}' not found", err=True)
        sys.exit(1)

    snapshot_path.unlink()
    click.echo(f"Deleted snapshot '{name}'")


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

    Example: okb sync run github --repo owner/repo
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
            click.echo(f"  ID {t.id} [{t.permissions}] {t.token_hash[:12]}...{desc}")
            created = t.created_at.strftime("%Y-%m-%d %H:%M")
            click.echo(f"      Created: {created}, Last used: {last_used}")
    except Exception as e:
        click.echo(f"Error listing tokens: {e}", err=True)
        sys.exit(1)


@token.command("revoke")
@click.argument("token_value", required=False)
@click.option("--id", "token_id", type=int, default=None, help="Token ID to revoke (from 'okb token list')")
@click.option("--db", "database", default=None, help="Database to revoke token from")
@click.pass_context
def token_revoke(ctx, token_value: str | None, token_id: int | None, database: str | None):
    """Revoke (delete) an API token.

    Either provide the full TOKEN_VALUE or use --id with the token ID from 'okb token list'.
    """
    from .tokens import delete_token, delete_token_by_id

    if not token_value and not token_id:
        click.echo("Error: Provide either TOKEN_VALUE or --id", err=True)
        sys.exit(1)

    if token_value and token_id:
        click.echo("Error: Provide either TOKEN_VALUE or --id, not both", err=True)
        sys.exit(1)

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    try:
        if token_id:
            deleted = delete_token_by_id(db_cfg.url, token_id)
            if deleted:
                click.echo(f"Token ID {token_id} revoked.")
            else:
                click.echo(f"Token ID {token_id} not found.", err=True)
                sys.exit(1)
        else:
            deleted = delete_token(db_cfg.url, token_value)
            if deleted:
                click.echo("Token revoked.")
            else:
                click.echo("Token not found. Use --id or provide the full token string.", err=True)
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
            click.echo("Backend: Modal GPU (deploy with: okb llm deploy)")
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

    Deploys a GPU-accelerated LLM service on Modal using the model from your config.
    Default: microsoft/Phi-3-mini-4k-instruct (no HuggingFace approval needed).

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

    # Get model and GPU from config
    model = config.llm_model or "microsoft/Phi-3-mini-4k-instruct"
    gpu = config.llm_modal_gpu or "L4"
    click.echo("Deploying Modal LLM:")
    click.echo(f"  Model: {model}")
    click.echo(f"  GPU: {gpu}")
    click.echo("Note: First deploy downloads the model and may take a few minutes.")

    # Set model and GPU in environment for Modal to pick up
    env = os.environ.copy()
    env["OKB_LLM_MODEL"] = model
    env["OKB_MODAL_GPU"] = gpu

    result = subprocess.run(
        ["modal", "deploy", str(llm_path)],
        cwd=llm_path.parent,
        env=env,
    )
    sys.exit(result.returncode)


# =============================================================================
# Enrich commands
# =============================================================================


@main.group()
def enrich():
    """LLM-based document enrichment (extract TODOs and entities)."""
    pass


@enrich.command("run")
@click.option("--db", "database", default=None, help="Database to enrich")
@click.option("--source-type", default=None, help="Filter by source type")
@click.option("--project", default=None, help="Filter by project")
@click.option("--query", default=None, help="Semantic search query to filter documents")
@click.option("--path-pattern", default=None, help="SQL LIKE pattern for source_path")
@click.option(
    "--all", "enrich_all", is_flag=True, help="Re-enrich all documents (ignore enriched_at)"
)
@click.option("--dry-run", is_flag=True, help="Show what would be enriched without executing")
@click.option("--limit", default=100, help="Maximum documents to process")
@click.option("--workers", default=None, type=int, help="Parallel workers (default: docs/5, min 1)")
@click.pass_context
def enrich_run(
    ctx,
    database: str | None,
    source_type: str | None,
    project: str | None,
    query: str | None,
    path_pattern: str | None,
    enrich_all: bool,
    dry_run: bool,
    limit: int,
    workers: int | None,
):
    """Run enrichment on documents to extract TODOs and entities.

    By default, only processes documents that haven't been enriched yet.
    Use --all to re-enrich all documents (e.g., after changing enrichment config).

    Examples:

        okb enrich run                  # Enrich un-enriched documents

        okb enrich run --dry-run        # Show what would be enriched

        okb enrich run --all            # Re-enrich everything

        okb enrich run --source-type markdown  # Only markdown files

        okb enrich run --query "meeting notes"  # Filter by semantic search

        okb enrich run --path-pattern '%myrepo%'  # Filter by source path

        okb enrich run --workers 8      # Use 8 parallel workers
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .llm import get_llm
    from .llm.enrich import EnrichmentConfig, get_unenriched_documents, process_enrichment

    # Check LLM is configured before doing any work
    if get_llm() is None:
        click.echo("Error: No LLM provider configured.", err=True)
        click.echo("", err=True)
        click.echo("Enrichment requires an LLM to extract TODOs and entities.", err=True)
        click.echo("Set ANTHROPIC_API_KEY or configure in ~/.config/okb/config.yaml:", err=True)
        click.echo("", err=True)
        click.echo("  llm:", err=True)
        click.echo("    provider: claude", err=True)
        click.echo("    model: claude-haiku-4-5-20251001", err=True)
        click.echo("", err=True)
        click.echo("Run 'okb llm status' to check configuration.", err=True)
        ctx.exit(1)

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    # Get enrichment version for re-enrichment check
    enrichment_version = config.enrichment_version if enrich_all else None

    click.echo(f"Scanning database '{db_cfg.name}' for documents to enrich...")
    if dry_run:
        click.echo("(dry run - no changes will be made)")

    docs = get_unenriched_documents(
        db_url=db_cfg.url,
        source_type=source_type,
        project=project,
        query=query,
        path_pattern=path_pattern,
        enrichment_version=enrichment_version,
        limit=limit,
    )

    if not docs:
        click.echo("No documents need enrichment.")
        return

    click.echo(f"Found {len(docs)} documents to enrich")

    if dry_run:
        for doc in docs[:20]:
            click.echo(f"  - {doc['title']} ({doc['source_type']})")
        if len(docs) > 20:
            click.echo(f"  ... and {len(docs) - 20} more")
        return

    # Calculate workers if not specified: floor(docs/5), minimum 1
    if workers is None:
        workers = max(1, len(docs) // 5)

    # Build config
    enrich_config = EnrichmentConfig.from_config(
        {
            "enabled": config.enrichment_enabled,
            "version": config.enrichment_version,
            "extract_todos": config.enrichment_extract_todos,
            "extract_entities": config.enrichment_extract_entities,
            "auto_create_todos": config.enrichment_auto_create_todos,
            "auto_create_entities": config.enrichment_auto_create_entities,
            "min_confidence_todo": config.enrichment_min_confidence_todo,
            "min_confidence_entity": config.enrichment_min_confidence_entity,
        }
    )

    total_todos = 0
    total_entities_pending = 0
    total_entities_created = 0
    completed = 0
    errors = 0

    def enrich_one(doc: dict) -> tuple[dict, dict | None, str | None]:
        """Process a single document. Returns (doc, stats, error)."""
        proj = doc["metadata"].get("project") if doc["metadata"] else None
        try:
            stats = process_enrichment(
                document_id=str(doc["id"]),
                source_path=doc["source_path"],
                title=doc["title"],
                content=doc["content"],
                source_type=doc["source_type"],
                db_url=db_cfg.url,
                config=enrich_config,
                project=proj,
            )
            return doc, stats, None
        except Exception as e:
            return doc, None, str(e)

    click.echo(f"Processing with {workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(enrich_one, doc): doc for doc in docs}

        for future in as_completed(futures):
            doc, stats, error = future.result()
            completed += 1
            title = doc["title"][:40] if doc["title"] else "Untitled"

            if error:
                errors += 1
                click.echo(f"[{completed}/{len(docs)}] {title}... -> error: {error[:50]}")
                continue

            total_todos += stats["todos_created"]
            total_entities_pending += stats["entities_pending"]
            total_entities_created += stats["entities_created"]

            parts = []
            if stats["todos_created"]:
                parts.append(f"{stats['todos_created']} TODOs")
            if stats["entities_pending"]:
                parts.append(f"{stats['entities_pending']} pending")
            if stats["entities_created"]:
                parts.append(f"{stats['entities_created']} entities")
            if parts:
                click.echo(f"[{completed}/{len(docs)}] {title}... -> {', '.join(parts)}")
            else:
                click.echo(f"[{completed}/{len(docs)}] {title}... -> nothing extracted")

    click.echo("")
    click.echo("Summary:")
    click.echo(f"  Documents processed: {len(docs)}")
    if errors:
        click.echo(f"  Errors: {errors}")
    click.echo(f"  TODOs created: {total_todos}")
    click.echo(f"  Entities pending review: {total_entities_pending}")
    click.echo(f"  Entities auto-created: {total_entities_created}")


@enrich.command("pending")
@click.option("--db", "database", default=None, help="Database to check")
@click.option("--type", "entity_type", default=None, help="Filter by entity type")
@click.option("--limit", default=50, help="Maximum results")
@click.pass_context
def enrich_pending(ctx, database: str | None, entity_type: str | None, limit: int):
    """List pending entity suggestions awaiting review.

    Shows entities extracted from documents that need approval before
    becoming searchable. Use 'okb enrich approve' or 'okb enrich reject'
    to process them.
    """
    from .llm.enrich import list_pending_entities

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    entities = list_pending_entities(db_cfg.url, entity_type=entity_type, limit=limit)

    if not entities:
        click.echo("No pending entity suggestions.")
        return

    click.echo(f"Pending entities ({len(entities)}):\n")
    for e in entities:
        confidence = e.get("confidence", 0)
        confidence_str = f" ({confidence:.0%})" if confidence else ""
        click.echo(f"  [{e['entity_type']}] {e['entity_name']}{confidence_str}")
        click.echo(f"    ID: {e['id']}")
        if e.get("description"):
            desc = (
                e["description"][:60] + "..."
                if len(e.get("description", "")) > 60
                else e["description"]
            )
            click.echo(f"    {desc}")
        if e.get("aliases"):
            click.echo(f"    Aliases: {', '.join(e['aliases'][:3])}")
        click.echo(f"    Source: {e['source_title']}")
        click.echo("")

    click.echo("Use 'okb enrich approve <id>' or 'okb enrich reject <id>' to process.")


@enrich.command("approve")
@click.argument("pending_id")
@click.option("--db", "database", default=None, help="Database")
@click.option("--local", is_flag=True, help="Use local CPU embedding instead of Modal")
@click.pass_context
def enrich_approve(ctx, pending_id: str, database: str | None, local: bool):
    """Approve a pending entity, creating it as a searchable document."""
    from .llm.enrich import approve_entity

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    source_path = approve_entity(db_cfg.url, pending_id, use_modal=not local)
    if source_path:
        click.echo(f"Entity approved and created: {source_path}")
    else:
        click.echo("Failed to approve entity. ID may be invalid or already processed.", err=True)
        sys.exit(1)


@enrich.command("reject")
@click.argument("pending_id")
@click.option("--db", "database", default=None, help="Database")
@click.pass_context
def enrich_reject(ctx, pending_id: str, database: str | None):
    """Reject a pending entity suggestion."""
    from .llm.enrich import reject_entity

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    if reject_entity(db_cfg.url, pending_id):
        click.echo("Entity rejected.")
    else:
        click.echo("Failed to reject entity. ID may be invalid or already processed.", err=True)
        sys.exit(1)


@enrich.command("analyze")
@click.option("--db", "database", default=None, help="Database to analyze")
@click.option("--project", default=None, help="Analyze specific project only")
@click.option("--sample-size", default=15, help="Number of documents to sample")
@click.option("--no-update", is_flag=True, help="Don't update database metadata")
@click.option("--stats-only", is_flag=True, help="Show stats without LLM analysis")
@click.pass_context
def enrich_analyze(
    ctx,
    database: str | None,
    project: str | None,
    sample_size: int,
    no_update: bool,
    stats_only: bool,
):
    """Analyze knowledge base and update description/topics.

    Uses entity aggregation and document sampling to understand the overall
    content and themes in the knowledge base. Generates a description and
    topic keywords using LLM analysis.

    Examples:

        okb enrich analyze              # Analyze entire database

        okb enrich analyze --stats-only # Show stats without LLM call

        okb enrich analyze --project myproject  # Analyze specific project

        okb enrich analyze --no-update  # Analyze without updating metadata
    """
    from .llm.analyze import (
        analyze_database,
        get_content_stats,
        get_entity_summary,
    )

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    scope = f"project '{project}'" if project else f"database '{db_cfg.name}'"
    click.echo(f"Analyzing {scope}...\n")

    # Always get stats
    stats = get_content_stats(db_cfg.url, project)
    entities = get_entity_summary(db_cfg.url, project, limit=20)

    # Show stats
    click.echo("Content Statistics:")
    click.echo(f"  Documents: {stats['total_documents']:,}")
    click.echo(f"  Tokens: ~{stats['total_tokens']:,}")
    if stats["source_types"]:
        sorted_types = sorted(stats["source_types"].items(), key=lambda x: -x[1])
        types_parts = [f"{t}: {c}" for t, c in sorted_types]
        # Break into multiple lines if many types
        if len(types_parts) > 4:
            click.echo("  Source types:")
            for tp in types_parts:
                click.echo(f"    {tp}")
        else:
            click.echo(f"  Source types: {', '.join(types_parts)}")
    if stats["projects"]:
        click.echo(f"  Projects: {', '.join(stats['projects'])}")
    if stats["date_range"]["earliest"]:
        earliest = stats["date_range"]["earliest"]
        latest = stats["date_range"]["latest"]
        click.echo(f"  Date range: {earliest} to {latest}")

    click.echo("")

    # Show top entities
    if entities:
        click.echo("Top Entities (by mentions):")
        for i, e in enumerate(entities[:10], 1):
            name, etype = e["name"], e["type"]
            refs, docs = e["ref_count"], e["doc_count"]
            click.echo(f"  {i}. {name} ({etype}) - {refs} mentions in {docs} docs")
        click.echo("")
    else:
        click.echo("No entities extracted yet.")
        click.echo("Run 'okb enrich run' to extract entities from documents.\n")

    if stats_only:
        return

    # Check LLM is configured
    from .llm import get_llm

    if get_llm() is None:
        click.echo("Error: No LLM provider configured.", err=True)
        click.echo("", err=True)
        click.echo("Analysis requires an LLM to generate description and topics.", err=True)
        click.echo("Set ANTHROPIC_API_KEY or configure in ~/.config/okb/config.yaml:", err=True)
        click.echo("", err=True)
        click.echo("  llm:", err=True)
        click.echo("    provider: claude", err=True)
        click.echo("", err=True)
        click.echo("Use --stats-only to see statistics without LLM analysis.", err=True)
        ctx.exit(1)

    click.echo(f"Sampling {sample_size} documents for analysis...")
    click.echo("Generating description and topics...")
    click.echo("")

    try:
        result = analyze_database(
            db_url=db_cfg.url,
            project=project,
            sample_size=sample_size,
            auto_update=not no_update,
        )

        click.echo("Analysis Complete:")
        click.echo(f"  Description: {result.description}")
        click.echo(f"  Topics: {', '.join(result.topics)}")

        if not no_update:
            click.echo("")
            click.echo("Updated database metadata.")
        else:
            click.echo("")
            click.echo("(metadata not updated - use without --no-update to save)")

    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        ctx.exit(1)


@enrich.command("consolidate")
@click.option("--db", "database", default=None, help="Database to consolidate")
@click.option("--duplicates/--no-duplicates", "detect_duplicates", default=True,
              help="Detect duplicate entities")
@click.option("--cross-doc/--no-cross-doc", "detect_cross_doc", default=True,
              help="Detect cross-document entities")
@click.option("--clusters/--no-clusters", "build_clusters", default=True,
              help="Build topic clusters")
@click.option("--relationships/--no-relationships", "extract_relationships", default=True,
              help="Extract entity relationships")
@click.option("--dry-run", is_flag=True, help="Show what would be found without creating proposals")
@click.pass_context
def enrich_consolidate(
    ctx,
    database: str | None,
    detect_duplicates: bool,
    detect_cross_doc: bool,
    build_clusters: bool,
    extract_relationships: bool,
    dry_run: bool,
):
    """Run entity consolidation pipeline.

    Detects duplicate entities, cross-document mentions, builds topic clusters,
    and extracts entity relationships. Creates pending proposals for review
    rather than auto-applying changes.

    Examples:

        okb enrich consolidate              # Run full consolidation

        okb enrich consolidate --dry-run    # Show what would be found

        okb enrich consolidate --no-clusters  # Skip clustering

        okb enrich consolidate --duplicates --no-cross-doc --no-clusters --no-relationships
    """
    from .llm import get_llm
    from .llm.consolidate import format_consolidation_result, run_consolidation

    # Check LLM is configured if needed
    if get_llm() is None:
        click.echo("Error: No LLM provider configured.", err=True)
        click.echo("Consolidation requires an LLM for deduplication and clustering.", err=True)
        click.echo("Set ANTHROPIC_API_KEY or configure in ~/.config/okb/config.yaml", err=True)
        ctx.exit(1)

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    click.echo(f"Running consolidation on database '{db_cfg.name}'...")
    if dry_run:
        click.echo("(dry run - no proposals will be created)")

    result = run_consolidation(
        db_url=db_cfg.url,
        detect_duplicates=detect_duplicates,
        detect_cross_doc=detect_cross_doc,
        build_clusters=build_clusters,
        extract_relationships=extract_relationships,
        dry_run=dry_run,
    )

    # Format and display result
    output = format_consolidation_result(result)
    click.echo("")
    click.echo(output)

    if not dry_run and (result.duplicates_found > 0 or result.cross_doc_candidates > 0):
        click.echo("")
        click.echo("Use 'okb enrich merge-proposals' to review pending merges.")


@enrich.command("merge-proposals")
@click.option("--db", "database", default=None, help="Database to check")
@click.option("--limit", default=50, help="Maximum results")
@click.pass_context
def enrich_merge_proposals(ctx, database: str | None, limit: int):
    """List pending entity merge proposals.

    Shows duplicate entities and cross-document mentions awaiting review.
    Use 'okb enrich approve-merge' or 'okb enrich reject-merge' to process.
    """
    from .llm.extractors.dedup import list_pending_merges

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    merges = list_pending_merges(db_cfg.url, limit=limit)

    if not merges:
        click.echo("No pending merge proposals.")
        return

    click.echo(f"Pending merge proposals ({len(merges)}):\n")
    for m in merges:
        confidence = m.get("confidence", 0)
        confidence_str = f" ({confidence:.0%})" if confidence else ""
        click.echo(f"  {m['canonical_name']} <- {m['duplicate_name']}{confidence_str}")
        click.echo(f"    ID: {m['id']}")
        click.echo(f"    Reason: {m.get('reason', 'similarity')}")
        click.echo("")

    click.echo("Use 'okb enrich approve-merge <id>' or 'okb enrich reject-merge <id>' to process.")


@enrich.command("approve-merge")
@click.argument("merge_id")
@click.option("--db", "database", default=None, help="Database")
@click.pass_context
def enrich_approve_merge(ctx, merge_id: str, database: str | None):
    """Approve a pending entity merge.

    Merges the duplicate entity into the canonical entity:
    - Redirects all entity references from duplicate to canonical
    - Adds duplicate's name as an alias for canonical
    - Deletes the duplicate entity document
    """
    from .llm.extractors.dedup import approve_merge

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    if approve_merge(db_cfg.url, merge_id):
        click.echo("Merge approved and executed.")
    else:
        click.echo("Failed to approve merge. ID may be invalid or already processed.", err=True)
        sys.exit(1)


@enrich.command("reject-merge")
@click.argument("merge_id")
@click.option("--db", "database", default=None, help="Database")
@click.pass_context
def enrich_reject_merge(ctx, merge_id: str, database: str | None):
    """Reject a pending entity merge proposal."""
    from .llm.extractors.dedup import reject_merge

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    if reject_merge(db_cfg.url, merge_id):
        click.echo("Merge rejected.")
    else:
        click.echo("Failed to reject merge. ID may be invalid or already processed.", err=True)
        sys.exit(1)


@enrich.command("clusters")
@click.option("--db", "database", default=None, help="Database to check")
@click.option("--limit", default=20, help="Maximum clusters to show")
@click.pass_context
def enrich_clusters(ctx, database: str | None, limit: int):
    """List topic clusters.

    Shows groups of related entities and documents organized by theme.
    """
    from .llm.consolidate import get_topic_clusters

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    clusters = get_topic_clusters(db_cfg.url, limit=limit)

    if not clusters:
        click.echo("No topic clusters found.")
        click.echo("Run 'okb enrich consolidate' to generate clusters.")
        return

    click.echo(f"Topic clusters ({len(clusters)}):\n")
    for c in clusters:
        click.echo(f"  {c['name']}")
        if c.get("description"):
            desc = c["description"][:70] + "..." if len(c["description"]) > 70 else c["description"]
            click.echo(f"    {desc}")
        click.echo(f"    Members: {c['member_count']} entities/documents")
        if c.get("sample_members"):
            samples = ", ".join(c["sample_members"][:5])
            click.echo(f"    Examples: {samples}")
        click.echo("")


@enrich.command("relationships")
@click.option("--db", "database", default=None, help="Database to check")
@click.option("--entity", "entity_name", default=None, help="Filter to specific entity")
@click.option("--type", "relationship_type", default=None,
              help="Filter by relationship type (works_for, uses, belongs_to, related_to)")
@click.option("--limit", default=50, help="Maximum results")
@click.pass_context
def enrich_relationships(
    ctx,
    database: str | None,
    entity_name: str | None,
    relationship_type: str | None,
    limit: int,
):
    """List entity relationships.

    Shows connections between entities (person→org, tech→project, etc.).

    Examples:

        okb enrich relationships                    # All relationships

        okb enrich relationships --entity "Django"  # Filter to one entity

        okb enrich relationships --type works_for   # Filter by type
    """
    from .llm.consolidate import get_entity_relationships

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    relationships = get_entity_relationships(
        db_cfg.url,
        entity_name=entity_name,
        relationship_type=relationship_type,
        limit=limit,
    )

    if not relationships:
        if entity_name:
            click.echo(f"No relationships found for entity '{entity_name}'.")
        else:
            click.echo("No relationships found.")
            click.echo("Run 'okb enrich consolidate' to extract relationships.")
        return

    click.echo(f"Entity relationships ({len(relationships)}):\n")
    for r in relationships:
        confidence = r.get("confidence", 0)
        conf_str = f" ({confidence:.0%})" if confidence else ""
        click.echo(f"  {r['source_name']} --[{r['relationship_type']}]--> {r['target_name']}{conf_str}")
        if r.get("evidence"):
            evidence = r["evidence"][:60] + "..." if len(r["evidence"]) > 60 else r["evidence"]
            click.echo(f"    Evidence: {evidence}")
    click.echo("")


@enrich.command("all")
@click.option("--db", "database", default=None, help="Database to enrich")
@click.option("--source-type", default=None, help="Filter by source type")
@click.option("--project", default=None, help="Filter by project")
@click.option("--query", default=None, help="Semantic search query to filter documents")
@click.option("--path-pattern", default=None, help="SQL LIKE pattern for source_path")
@click.option("--limit", default=100, help="Maximum documents to process")
@click.option("--workers", default=None, type=int, help="Parallel workers (default: docs/5, min 1)")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option("--skip-consolidate", is_flag=True, help="Skip consolidation phase")
@click.option("--duplicates/--no-duplicates", "detect_duplicates", default=True,
              help="Detect duplicate entities during consolidation")
@click.option("--clusters/--no-clusters", "build_clusters", default=True,
              help="Build topic clusters during consolidation")
@click.option("--relationships/--no-relationships", "extract_relationships", default=True,
              help="Extract entity relationships during consolidation")
@click.pass_context
def enrich_all(
    ctx,
    database: str | None,
    source_type: str | None,
    project: str | None,
    query: str | None,
    path_pattern: str | None,
    limit: int,
    workers: int | None,
    dry_run: bool,
    skip_consolidate: bool,
    detect_duplicates: bool,
    build_clusters: bool,
    extract_relationships: bool,
):
    """Run full enrichment pipeline: extraction + consolidation.

    Combines 'enrich run' and 'enrich consolidate' in one command for
    one-shot enrichment of documents.

    Examples:

        okb enrich all                          # Run full pipeline

        okb enrich all --dry-run                # Preview what would happen

        okb enrich all --skip-consolidate       # Run extraction only

        okb enrich all --source-type markdown   # Filter to markdown files

        okb enrich all --no-clusters            # Skip cluster building
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .llm import get_llm
    from .llm.consolidate import format_consolidation_result, run_consolidation
    from .llm.enrich import EnrichmentConfig, get_unenriched_documents, process_enrichment

    # Check LLM is configured
    if get_llm() is None:
        click.echo("Error: No LLM provider configured.", err=True)
        click.echo("Set ANTHROPIC_API_KEY or configure in ~/.config/okb/config.yaml", err=True)
        ctx.exit(1)

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    # Phase 1: Enrichment
    click.echo("=== Phase 1: Enrichment ===")
    click.echo(f"Scanning database '{db_cfg.name}' for documents to enrich...")
    if dry_run:
        click.echo("(dry run - no changes will be made)")

    docs = get_unenriched_documents(
        db_url=db_cfg.url,
        source_type=source_type,
        project=project,
        query=query,
        path_pattern=path_pattern,
        limit=limit,
    )

    total_todos = 0
    total_entities_pending = 0
    total_entities_created = 0

    if not docs:
        click.echo("No documents need enrichment.")
    else:
        click.echo(f"Found {len(docs)} documents to enrich")

        if dry_run:
            for doc in docs[:20]:
                click.echo(f"  - {doc['title']} ({doc['source_type']})")
            if len(docs) > 20:
                click.echo(f"  ... and {len(docs) - 20} more")
        else:
            # Build config
            enrich_config = EnrichmentConfig.from_config(
                {
                    "enabled": config.enrichment_enabled,
                    "version": config.enrichment_version,
                    "extract_todos": config.enrichment_extract_todos,
                    "extract_entities": config.enrichment_extract_entities,
                    "auto_create_todos": config.enrichment_auto_create_todos,
                    "auto_create_entities": config.enrichment_auto_create_entities,
                    "min_confidence_todo": config.enrichment_min_confidence_todo,
                    "min_confidence_entity": config.enrichment_min_confidence_entity,
                }
            )

            # Calculate workers
            if workers is None:
                workers = max(1, len(docs) // 5)

            completed = 0
            errors = 0

            def enrich_one(doc: dict) -> tuple[dict, dict | None, str | None]:
                proj = doc["metadata"].get("project") if doc["metadata"] else None
                try:
                    stats = process_enrichment(
                        document_id=str(doc["id"]),
                        source_path=doc["source_path"],
                        title=doc["title"],
                        content=doc["content"],
                        source_type=doc["source_type"],
                        db_url=db_cfg.url,
                        config=enrich_config,
                        project=proj,
                    )
                    return doc, stats, None
                except Exception as e:
                    return doc, None, str(e)

            click.echo(f"Processing with {workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(enrich_one, doc): doc for doc in docs}

                for future in as_completed(futures):
                    doc, stats, error = future.result()
                    completed += 1
                    title = doc["title"][:40] if doc["title"] else "Untitled"

                    if error:
                        errors += 1
                        click.echo(f"[{completed}/{len(docs)}] {title}... -> error: {error[:50]}")
                        continue

                    total_todos += stats["todos_created"]
                    total_entities_pending += stats["entities_pending"]
                    total_entities_created += stats["entities_created"]

                    parts = []
                    if stats["todos_created"]:
                        parts.append(f"{stats['todos_created']} TODOs")
                    if stats["entities_pending"]:
                        parts.append(f"{stats['entities_pending']} pending")
                    if stats["entities_created"]:
                        parts.append(f"{stats['entities_created']} entities")
                    if parts:
                        click.echo(f"[{completed}/{len(docs)}] {title}... -> {', '.join(parts)}")
                    else:
                        click.echo(f"[{completed}/{len(docs)}] {title}... -> nothing extracted")

            click.echo("")
            click.echo("Enrichment summary:")
            click.echo(f"  Documents processed: {len(docs)}")
            if errors:
                click.echo(f"  Errors: {errors}")
            click.echo(f"  TODOs created: {total_todos}")
            click.echo(f"  Entities pending review: {total_entities_pending}")
            click.echo(f"  Entities auto-created: {total_entities_created}")

    # Phase 2: Consolidation
    if skip_consolidate:
        click.echo("")
        click.echo("Skipping consolidation (--skip-consolidate)")
        return

    click.echo("")
    click.echo("=== Phase 2: Consolidation ===")

    result = run_consolidation(
        db_url=db_cfg.url,
        detect_duplicates=detect_duplicates,
        detect_cross_doc=True,
        build_clusters=build_clusters,
        extract_relationships=extract_relationships,
        dry_run=dry_run,
    )

    output = format_consolidation_result(result)
    click.echo(output)

    if not dry_run and (result.duplicates_found > 0 or result.cross_doc_candidates > 0):
        click.echo("")
        click.echo("Use 'okb enrich review' to review pending entities and merges.")


@enrich.command("review")
@click.option("--db", "database", default=None, help="Database to review")
@click.option("--entities-only", is_flag=True, help="Only review pending entities")
@click.option("--merges-only", is_flag=True, help="Only review pending merges")
@click.option("--local", is_flag=True, help="Use local CPU embedding instead of Modal")
@click.option("--wait/--no-wait", default=True, help="Wait for embeddings to complete")
@click.pass_context
def enrich_review(
    ctx, database: str | None, entities_only: bool, merges_only: bool, local: bool, wait: bool
):
    """Interactive review of pending entities and merge proposals.

    Loops through pending items with approve/reject prompts.
    Press Q to quit early - remaining items stay pending for later.

    Entity approvals run asynchronously - you can continue reviewing while
    embeddings are generated. Use --no-wait to exit immediately after reviewing.

    Examples:

        okb enrich review                    # Review all pending items

        okb enrich review --entities-only    # Only review entities

        okb enrich review --merges-only      # Only review merges

        okb enrich review --local            # Use local CPU embedding

        okb enrich review --no-wait          # Don't wait for embeddings
    """

    from .llm.enrich import (
        approve_entity_async,
        list_pending_entities,
        reject_entity,
        shutdown_executor,
    )
    from .llm.extractors.dedup import approve_merge, list_pending_merges, reject_merge

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)
    use_modal = not local

    # Get pending items
    entities = [] if merges_only else list_pending_entities(db_cfg.url, limit=100)
    merges = [] if entities_only else list_pending_merges(db_cfg.url, limit=100)

    if not entities and not merges:
        click.echo("No pending items to review.")
        return

    click.echo(f"Pending: {len(entities)} entities, {len(merges)} merges")
    click.echo("")

    # Counters
    approved = 0
    rejected = 0
    skipped = 0

    # Track async approval futures
    pending_futures: list[tuple] = []  # (future, entity_name)

    # Review entities
    choice = None
    if entities and not merges_only:
        for i, e in enumerate(entities, 1):
            # Check for completed futures
            done_count = sum(1 for f, _ in pending_futures if f.done())
            if pending_futures and done_count > 0:
                total = len(pending_futures)
                click.echo(click.style(f"  ({done_count}/{total} embeddings done)", dim=True))

            click.echo(click.style(f"=== Entity Review [{i}/{len(entities)}] ===", bold=True))
            click.echo(f"Name: {click.style(e['entity_name'], fg='cyan')}")
            click.echo(f"Type: {e['entity_type']}")
            confidence = e.get("confidence", 0)
            if confidence:
                click.echo(f"Confidence: {confidence:.0%}")
            if e.get("description"):
                d = e["description"]
                desc = d[:80] + "..." if len(d) > 80 else d
                click.echo(f"Description: {desc}")
            if e.get("aliases"):
                click.echo(f"Aliases: {', '.join(e['aliases'][:5])}")
            click.echo(f"Source: {e['source_title']}")
            click.echo("")

            choice = click.prompt(
                "[A]pprove  [R]eject  [S]kip  [Q]uit",
                type=click.Choice(["A", "R", "S", "Q", "a", "r", "s", "q"]),
                show_choices=False,
            ).upper()

            if choice == "Q":
                click.echo("Quitting review...")
                break
            elif choice == "A":
                # Submit async approval
                future = approve_entity_async(db_cfg.url, str(e["id"]), use_modal)
                pending_futures.append((future, e["entity_name"]))
                click.echo(click.style("⏳ Queued for approval", fg="cyan"))
                approved += 1
            elif choice == "R":
                if reject_entity(db_cfg.url, str(e["id"])):
                    click.echo(click.style("✗ Rejected", fg="yellow"))
                    rejected += 1
                else:
                    click.echo(click.style("✗ Failed to reject", fg="red"))
            else:
                click.echo("Skipped")
                skipped += 1

            click.echo("")
        else:
            # Completed all entities, continue to merges
            pass

    # Review merges (only if we didn't quit early)
    if merges and not entities_only and (not entities or choice != "Q"):
        for i, m in enumerate(merges, 1):
            click.echo(click.style(f"=== Merge Review [{i}/{len(merges)}] ===", bold=True))
            cname = click.style(m["canonical_name"], fg="cyan")
            ctype = m.get("canonical_type", "unknown")
            click.echo(f"Canonical: {cname} ({ctype})")
            dname = click.style(m["duplicate_name"], fg="yellow")
            dtype = m.get("duplicate_type", "unknown")
            click.echo(f"Duplicate: {dname} ({dtype})")
            confidence = m.get("confidence", 0)
            if confidence:
                click.echo(f"Confidence: {confidence:.0%}")
            click.echo(f"Reason: {m.get('reason', 'similarity')}")
            click.echo("")

            choice = click.prompt(
                "[A]pprove  [R]eject  [S]kip  [Q]uit",
                type=click.Choice(["A", "R", "S", "Q", "a", "r", "s", "q"]),
                show_choices=False,
            ).upper()

            if choice == "Q":
                click.echo("Quitting review...")
                break
            elif choice == "A":
                if approve_merge(db_cfg.url, str(m["id"])):
                    click.echo(click.style("✓ Merged", fg="green"))
                    approved += 1
                else:
                    click.echo(click.style("✗ Failed to merge", fg="red"))
            elif choice == "R":
                if reject_merge(db_cfg.url, str(m["id"])):
                    click.echo(click.style("✗ Rejected", fg="yellow"))
                    rejected += 1
                else:
                    click.echo(click.style("✗ Failed to reject", fg="red"))
            else:
                click.echo("Skipped")
                skipped += 1

            click.echo("")

    # Wait for pending approvals if requested
    if pending_futures:
        if wait:
            click.echo(f"Waiting for {len(pending_futures)} pending approvals...")
            succeeded = 0
            failed = 0
            for future, name in pending_futures:
                try:
                    result = future.result(timeout=120)
                    if result:
                        click.echo(click.style(f"  ✓ {name}", fg="green"))
                        succeeded += 1
                    else:
                        click.echo(click.style(f"  ✗ {name} failed", fg="red"))
                        failed += 1
                except Exception as e:
                    click.echo(click.style(f"  ✗ {name}: {e}", fg="red"))
                    failed += 1
            click.echo(f"Embeddings: {succeeded} succeeded, {failed} failed")
        else:
            done_count = sum(1 for f, _ in pending_futures if f.done())
            pending_count = len(pending_futures) - done_count
            if pending_count > 0:
                click.echo(f"{pending_count} embeddings still processing in background...")

    # Cleanup executor
    shutdown_executor(wait=wait)

    # Summary
    click.echo("")
    click.echo(click.style("Review complete:", bold=True))
    click.echo(f"  {click.style(str(approved), fg='green')} approved")
    click.echo(f"  {click.style(str(rejected), fg='yellow')} rejected")
    click.echo(f"  {skipped} skipped")


if __name__ == "__main__":
    main()
