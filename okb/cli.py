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

from .config import (
    InsecureConfigError,
    check_file_permissions,
    config,
    get_config_dir,
    get_config_path,
    get_default_config_yaml,
)


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
    config_path.chmod(0o600)
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
@click.option("--channel", multiple=True, help="Slack channel ID to sync (can repeat)")
@click.option("--repo", multiple=True, help="GitHub repo to sync (owner/repo, can repeat)")
@click.option(
    "--source", "include_source", is_flag=True, help="Sync all source files (not just README+docs)"
)
@click.option("--issues", "include_issues", is_flag=True, help="Include GitHub issues")
@click.option("--prs", "include_prs", is_flag=True, help="Include GitHub pull requests")
@click.option("--wiki", "include_wiki", is_flag=True, help="Include GitHub wiki pages")
@click.option("--branch", default=None, help="Git branch to sync (default: repo default)")
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
    channel: tuple[str, ...],
    repo: tuple[str, ...],
    include_source: bool,
    include_issues: bool,
    include_prs: bool,
    include_wiki: bool,
    branch: str | None,
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
        source_names = config.list_enabled_sources(db_cfg.name)
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

            # Get and resolve config (per-db sources override global)
            source_cfg = config.get_source_config(source_name, db_cfg.name)
            if source_cfg is None:
                click.echo(f"Skipping '{source_name}': not configured or disabled", err=True)
                continue

            # Merge CLI options into config (override config file values)
            if folder:
                source_cfg["folders"] = list(folder)
            if doc_ids:
                source_cfg["doc_ids"] = list(doc_ids)
            # Slack-specific options
            if channel:
                source_cfg["channels"] = list(channel)
            # GitHub-specific options
            if repo:
                source_cfg["repos"] = list(repo)
            if branch:
                source_cfg["branch"] = branch
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
@click.option("--db", "database", default=None, help="Show sources for specific database")
@click.pass_context
def sync_list(ctx, database: str | None):
    """List available API sources."""
    from .plugins.registry import PluginRegistry

    db_name = database or ctx.obj.get("database")
    installed = PluginRegistry.list_sources()
    configured = config.list_enabled_sources(db_name)

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
@click.option("--db", "database", default=None, help="Use source config from specific database")
@click.pass_context
def sync_list_projects(ctx, source: str, database: str | None):
    """List projects from an API source (for finding project IDs).

    Example: okb sync list-projects todoist
    """
    from .plugins.registry import PluginRegistry

    db_name = database or ctx.obj.get("database")

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

    # Get and resolve config (per-db sources override global)
    source_cfg = config.get_source_config(source, db_name)
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
# Synthesize commands
# =============================================================================


@main.group()
def synthesize():
    """LLM-based knowledge synthesis (generate topic summaries and insights)."""
    pass


@synthesize.command("run")
@click.option("--db", "database", default=None, help="Database to synthesize from")
@click.option("--project", default=None, help="Scope to specific project")
@click.option("--sample-size", default=25, help="Number of documents to sample")
@click.option("--max-proposals", default=10, help="Maximum proposals to generate")
@click.option("--dry-run", is_flag=True, help="Show what would be sampled without calling LLM")
@click.pass_context
def synthesize_run(
    ctx,
    database: str | None,
    project: str | None,
    sample_size: int,
    max_proposals: int,
    dry_run: bool,
):
    """Run synthesis to propose knowledge documents.

    Samples the database broadly and uses LLM to propose topic summaries,
    entity profiles, relationship maps, and cross-cutting insights.

    Examples:

        okb synthesize run                      # Generate proposals

        okb synthesize run --dry-run            # Preview what would be sampled

        okb synthesize run --project myproject  # Scope to project

        okb synthesize run --max-proposals 5    # Fewer proposals
    """
    from .llm import get_llm
    from .llm.analyze import get_content_stats, get_document_samples

    if get_llm() is None:
        click.echo("Error: No LLM provider configured.", err=True)
        click.echo("Set ANTHROPIC_API_KEY or configure in ~/.config/okb/config.yaml", err=True)
        ctx.exit(1)

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    scope = f"project '{project}'" if project else f"database '{db_cfg.name}'"
    click.echo(f"Synthesizing from {scope}...")

    if dry_run:
        stats = get_content_stats(db_cfg.url, project)
        samples = get_document_samples(db_cfg.url, project, sample_size, strategy="diverse")
        click.echo(f"Would sample {len(samples)} of {stats['total_documents']} documents:")
        for s in samples[:15]:
            title = s["title"][:60] if s["title"] else "Untitled"
            click.echo(f"  - {title} ({s['source_type']})")
        if len(samples) > 15:
            click.echo(f"  ... and {len(samples) - 15} more")
        return

    from .llm.synthesize import synthesize

    try:
        result = synthesize(
            db_url=db_cfg.url,
            project=project,
            sample_size=sample_size,
            max_proposals=max_proposals,
            origin="cli",
        )

        if not result.proposals:
            click.echo("No proposals generated. The LLM response could not be parsed.")
            if result.raw_response:
                click.echo(f"\nRaw LLM response (first 500 chars):\n{result.raw_response[:500]}")
            return

        click.echo(f"\nCreated {result.proposals_created} proposals (model: {result.model}):\n")
        for p in result.proposals:
            click.echo(f"  - {p['title']}")
            click.echo(f"    ID: {p['id']}")

        click.echo("\nUse 'okb synthesize review' to approve/reject/edit proposals.")

    except Exception as e:
        click.echo(f"Error during synthesis: {e}", err=True)
        ctx.exit(1)


@synthesize.command("pending")
@click.option("--db", "database", default=None, help="Database to check")
@click.option("--project", default=None, help="Filter by project")
@click.pass_context
def synthesize_pending(ctx, database: str | None, project: str | None):
    """List pending synthesis proposals.

    Shows proposals awaiting review. Use 'okb synthesize review' for
    interactive approval, or approve/reject individually.
    """
    from .llm.synthesize import list_pending_synthesis

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    proposals = list_pending_synthesis(db_cfg.url, project=project)

    if not proposals:
        click.echo("No pending synthesis proposals.")
        return

    click.echo(f"Pending proposals ({len(proposals)}):\n")
    for p in proposals:
        click.echo(f"  {p['title']}")
        click.echo(f"    ID: {p['id']}")
        excerpt = p.get("excerpt", "")
        if excerpt:
            click.echo(f"    {excerpt}...")
        click.echo(f"    Model: {p['model']} | Origin: {p['origin']}")
        click.echo("")

    click.echo("Use 'okb synthesize review' or 'okb synthesize approve <id>' to process.")


@synthesize.command("approve")
@click.argument("pending_id")
@click.option("--db", "database", default=None, help="Database")
@click.option("--title", default=None, help="Override title")
@click.option("--content", default=None, help="Override content")
@click.pass_context
def synthesize_approve(
    ctx, pending_id: str, database: str | None, title: str | None, content: str | None
):
    """Approve a pending synthesis proposal, creating a searchable document."""
    from .llm.synthesize import approve_synthesis

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    source_path = approve_synthesis(db_cfg.url, pending_id, title=title, content=content)
    if source_path:
        click.echo(f"Synthesis approved and created: {source_path}")
    else:
        click.echo("Failed to approve. ID may be invalid or already processed.", err=True)
        sys.exit(1)


@synthesize.command("reject")
@click.argument("pending_id")
@click.option("--db", "database", default=None, help="Database")
@click.pass_context
def synthesize_reject(ctx, pending_id: str, database: str | None):
    """Reject a pending synthesis proposal."""
    from .llm.synthesize import reject_synthesis

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    if reject_synthesis(db_cfg.url, pending_id):
        click.echo("Synthesis proposal rejected.")
    else:
        click.echo("Failed to reject. ID may be invalid or already processed.", err=True)
        sys.exit(1)


@synthesize.command("review")
@click.option("--db", "database", default=None, help="Database to review")
@click.option("--project", default=None, help="Filter by project")
@click.option("--wait/--no-wait", default=True, help="Wait for embeddings to complete")
@click.pass_context
def synthesize_review(ctx, database: str | None, project: str | None, wait: bool):
    """Interactive review of pending synthesis proposals.

    Loops through proposals with approve/edit/reject/skip prompts.
    Press Q to quit early - remaining items stay pending for later.

    Approvals run asynchronously - you can continue reviewing while
    embeddings are generated. Use --no-wait to exit immediately.

    Examples:

        okb synthesize review              # Review all pending

        okb synthesize review --project x  # Review specific project
    """
    from .llm.synthesize import (
        approve_synthesis_async,
        get_pending_synthesis,
        list_pending_synthesis,
        reject_synthesis,
        shutdown_executor,
        update_pending_synthesis,
    )

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    proposals = list_pending_synthesis(db_cfg.url, project=project, limit=100)

    if not proposals:
        click.echo("No pending synthesis proposals to review.")
        return

    click.echo(f"Pending: {len(proposals)} proposals")
    click.echo("")

    approved = 0
    rejected = 0
    skipped = 0
    pending_futures: list[tuple] = []

    for i, p in enumerate(proposals, 1):
        # Check for completed futures
        done_count = sum(1 for f, _ in pending_futures if f.done())
        if pending_futures and done_count > 0:
            total = len(pending_futures)
            click.echo(click.style(f"  ({done_count}/{total} embeddings done)", dim=True))

        # Get full content for display
        full = get_pending_synthesis(db_cfg.url, str(p["id"]))
        if not full or full["status"] != "pending":
            continue

        click.echo(click.style(f"=== Proposal [{i}/{len(proposals)}] ===", bold=True))
        click.echo(f"Title: {click.style(full['title'], fg='cyan')}")
        click.echo(f"Model: {full['model']}")
        click.echo("")
        # Show content preview (first ~300 chars)
        content_preview = full["content"][:300]
        if len(full["content"]) > 300:
            content_preview += "..."
        click.echo(content_preview)
        click.echo("")

        choice = click.prompt(
            "[A]pprove  [E]dit  [R]eject  [S]kip  [Q]uit",
            type=click.Choice(["A", "E", "R", "S", "Q", "a", "e", "r", "s", "q"]),
            show_choices=False,
        ).upper()

        if choice == "Q":
            click.echo("Quitting review...")
            break
        elif choice == "E":
            # Open in editor
            edited_text = f"# {full['title']}\n\n{full['content']}"
            result = click.edit(edited_text)
            if result:
                # Parse: first line (after #) is title, rest is content
                lines = result.strip().split("\n")
                new_title = lines[0].lstrip("# ").strip() if lines else full["title"]
                new_content = "\n".join(lines[1:]).strip() if len(lines) > 1 else full["content"]
                if new_title and new_content:
                    update_pending_synthesis(
                        db_cfg.url, str(full["id"]),
                        title=new_title, content=new_content,
                    )
                    click.echo(click.style("Updated. Approving...", fg="cyan"))
                    future = approve_synthesis_async(
                        db_cfg.url, str(full["id"]),
                        title=new_title, content=new_content,
                    )
                    pending_futures.append((future, new_title))
                    approved += 1
                else:
                    click.echo("Edit was empty, skipping.")
                    skipped += 1
            else:
                click.echo("No changes, skipping.")
                skipped += 1
        elif choice == "A":
            future = approve_synthesis_async(db_cfg.url, str(full["id"]))
            pending_futures.append((future, full["title"]))
            click.echo(click.style("Queued for approval", fg="cyan"))
            approved += 1
        elif choice == "R":
            if reject_synthesis(db_cfg.url, str(full["id"])):
                click.echo(click.style("Rejected", fg="yellow"))
                rejected += 1
            else:
                click.echo(click.style("Failed to reject", fg="red"))
        else:
            click.echo("Skipped")
            skipped += 1

        click.echo("")

    # Wait for pending approvals
    if pending_futures:
        if wait:
            click.echo(f"Waiting for {len(pending_futures)} pending approvals...")
            succeeded = 0
            failed = 0
            for future, title in pending_futures:
                try:
                    result = future.result(timeout=120)
                    if result:
                        click.echo(click.style(f"  Created: {title}", fg="green"))
                        succeeded += 1
                    else:
                        click.echo(click.style(f"  Failed: {title}", fg="red"))
                        failed += 1
                except Exception as e:
                    click.echo(click.style(f"  Error: {title}: {e}", fg="red"))
                    failed += 1
            click.echo(f"Embeddings: {succeeded} succeeded, {failed} failed")
        else:
            done_count = sum(1 for f, _ in pending_futures if f.done())
            pending_count = len(pending_futures) - done_count
            if pending_count > 0:
                click.echo(f"{pending_count} embeddings still processing in background...")

    shutdown_executor(wait=wait)

    click.echo("")
    click.echo(click.style("Review complete:", bold=True))
    click.echo(f"  {click.style(str(approved), fg='green')} approved")
    click.echo(f"  {click.style(str(rejected), fg='yellow')} rejected")
    click.echo(f"  {skipped} skipped")


@synthesize.command("analyze")
@click.option("--db", "database", default=None, help="Database to analyze")
@click.option("--project", default=None, help="Analyze specific project only")
@click.option("--sample-size", default=15, help="Number of documents to sample")
@click.option("--no-update", is_flag=True, help="Don't update database metadata")
@click.option("--stats-only", is_flag=True, help="Show stats without LLM analysis")
@click.pass_context
def synthesize_analyze(
    ctx,
    database: str | None,
    project: str | None,
    sample_size: int,
    no_update: bool,
    stats_only: bool,
):
    """Analyze knowledge base and update description/topics.

    Uses document sampling to understand the overall content and themes.
    Generates a description and topic keywords using LLM analysis.

    Examples:

        okb synthesize analyze              # Analyze entire database

        okb synthesize analyze --stats-only # Show stats without LLM call

        okb synthesize analyze --project p  # Analyze specific project

        okb synthesize analyze --no-update  # Analyze without updating metadata
    """
    from .llm.analyze import analyze_database, get_content_stats

    db_name = database or ctx.obj.get("database")
    db_cfg = config.get_database(db_name)

    scope = f"project '{project}'" if project else f"database '{db_cfg.name}'"
    click.echo(f"Analyzing {scope}...\n")

    stats = get_content_stats(db_cfg.url, project)

    click.echo("Content Statistics:")
    click.echo(f"  Documents: {stats['total_documents']:,}")
    click.echo(f"  Tokens: ~{stats['total_tokens']:,}")
    if stats["source_types"]:
        sorted_types = sorted(stats["source_types"].items(), key=lambda x: -x[1])
        types_parts = [f"{t}: {c}" for t, c in sorted_types]
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

    if stats_only:
        return

    from .llm import get_llm

    if get_llm() is None:
        click.echo("Error: No LLM provider configured.", err=True)
        click.echo("Set ANTHROPIC_API_KEY or configure in ~/.config/okb/config.yaml", err=True)
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


# =============================================================================
# Service commands (systemd user services)
# =============================================================================


def _get_systemd_user_dir() -> Path:
    """Get systemd user service directory."""
    return Path.home() / ".config" / "systemd" / "user"


def _get_okb_path() -> str | None:
    """Get the path to the okb-admin executable."""
    return shutil.which("okb-admin")


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run systemctl --user command."""
    cmd = ["systemctl", "--user"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _generate_db_service(okb_path: str) -> str:
    """Generate okb-db.service content."""
    return f"""[Unit]
Description=OKB Database Container

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart={okb_path} db start
ExecStop={okb_path} db stop

[Install]
WantedBy=default.target
"""


def _get_env_file_path() -> Path:
    """Get the path to the environment file for systemd services."""
    config_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "okb"
    return config_dir / "env"


def _generate_http_service(okb_path: str) -> str:
    """Generate okb.service content."""
    # Build a minimal PATH with essential directories
    # Filter out problematic paths (WSL Windows paths with spaces, etc.)
    current_path = os.environ.get("PATH", "")
    essential_paths = []
    for p in current_path.split(":"):
        # Skip Windows paths in WSL (contain spaces, not needed)
        if p.startswith("/mnt/"):
            continue
        # Skip paths with spaces (would break systemd Environment line)
        if " " in p:
            continue
        # Keep paths that exist
        if Path(p).is_dir():
            essential_paths.append(p)

    # Ensure we have basic system paths
    for fallback in ["/usr/local/bin", "/usr/bin", "/bin"]:
        if fallback not in essential_paths and Path(fallback).is_dir():
            essential_paths.append(fallback)

    path_dirs = ":".join(essential_paths)
    env_file = _get_env_file_path()

    return f"""[Unit]
Description=OKB Knowledge Base HTTP Server
After=okb-db.service
Requires=okb-db.service

[Service]
Type=simple
ExecStart={okb_path} serve --http
Restart=on-failure
RestartSec=5
Environment=PATH={path_dirs}
EnvironmentFile=-{env_file}

[Install]
WantedBy=default.target
"""


@main.group()
def service():
    """Manage okb systemd user services."""
    pass


@service.command("install")
@click.option("--no-start", is_flag=True, help="Don't enable/start services after install")
def service_install(no_start: bool):
    """Install systemd user services for okb.

    Creates two systemd user services:
      - okb-db.service: Manages the Docker container
      - okb.service: HTTP server (depends on okb-db)

    After installation, services are enabled and started by default.
    """
    okb_path = _get_okb_path()
    if not okb_path:
        click.echo("Error: okb-admin executable not found in PATH", err=True)
        sys.exit(1)

    systemd_dir = _get_systemd_user_dir()
    systemd_dir.mkdir(parents=True, exist_ok=True)

    db_service_path = systemd_dir / "okb-db.service"
    http_service_path = systemd_dir / "okb.service"

    # Create environment file template if it doesn't exist
    env_file = _get_env_file_path()
    if not env_file.exists():
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            "# Environment variables for OKB systemd services.\n"
            "# Uncomment and set values as needed.\n"
            "#ANTHROPIC_API_KEY=sk-ant-...\n"
            "#TODOIST_TOKEN=\n"
            "#GITHUB_TOKEN=\n"
        )
        env_file.chmod(0o600)
        click.echo(f"Created environment file: {env_file}")
        click.echo("  Edit this file to set API keys for the service.")
    else:
        check_file_permissions(env_file)

    # Write service files
    click.echo(f"Installing service files to {systemd_dir}")

    db_service_content = _generate_db_service(okb_path)
    db_service_path.write_text(db_service_content)
    click.echo(f"  Created {db_service_path.name}")

    http_service_content = _generate_http_service(okb_path)
    http_service_path.write_text(http_service_content)
    click.echo(f"  Created {http_service_path.name}")

    # Reload systemd
    click.echo("Reloading systemd...")
    result = _systemctl("daemon-reload", check=False)
    if result.returncode != 0:
        click.echo(f"Warning: daemon-reload failed: {result.stderr}", err=True)

    if not no_start:
        # Enable and start services
        click.echo("Enabling and starting services...")
        result = _systemctl("enable", "--now", "okb-db", "okb", check=False)
        if result.returncode != 0:
            click.echo(f"Warning: failed to enable/start services: {result.stderr}", err=True)
        else:
            click.echo("Services started.")

    click.echo("")
    click.echo("Installation complete.")
    click.echo("")
    click.echo("To persist services after logout, run:")
    click.echo("  sudo loginctl enable-linger $USER")


@service.command("uninstall")
def service_uninstall():
    """Remove systemd user services.

    Stops and disables the services, then removes the service files.
    """
    systemd_dir = _get_systemd_user_dir()
    db_service_path = systemd_dir / "okb-db.service"
    http_service_path = systemd_dir / "okb.service"

    # Stop services if running
    click.echo("Stopping services...")
    _systemctl("stop", "okb", "okb-db", check=False)

    # Disable services
    click.echo("Disabling services...")
    _systemctl("disable", "okb", "okb-db", check=False)

    # Remove service files
    removed = []
    for path in [http_service_path, db_service_path]:
        if path.exists():
            path.unlink()
            removed.append(path.name)
            click.echo(f"  Removed {path.name}")

    if not removed:
        click.echo("No service files found.")
        return

    # Reload systemd
    click.echo("Reloading systemd...")
    _systemctl("daemon-reload", check=False)

    click.echo("Uninstallation complete.")


@service.command("status")
def service_status():
    """Show status of okb services."""
    # Check if service files exist
    systemd_dir = _get_systemd_user_dir()
    db_service_path = systemd_dir / "okb-db.service"
    http_service_path = systemd_dir / "okb.service"

    if not db_service_path.exists() and not http_service_path.exists():
        click.echo("okb services are not installed.")
        click.echo("Run 'okb service install' to install them.")
        return

    # Show status for both services
    result = subprocess.run(
        ["systemctl", "--user", "status", "okb-db", "okb", "--no-pager"],
        capture_output=False,
        check=False,
    )
    sys.exit(result.returncode)


@service.command("start")
def service_start():
    """Start okb services."""
    click.echo("Starting okb services...")
    result = _systemctl("start", "okb-db", "okb", check=False)
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Services started.")


@service.command("stop")
def service_stop():
    """Stop okb services."""
    click.echo("Stopping okb services...")
    result = _systemctl("stop", "okb", "okb-db", check=False)
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Services stopped.")


@service.command("restart")
def service_restart():
    """Restart okb services.

    Use this after upgrading okb to pick up code changes.
    """
    click.echo("Restarting okb services...")
    result = _systemctl("restart", "okb-db", "okb", check=False)
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Services restarted.")


@service.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
@click.option("-n", "--lines", default=50, help="Number of lines to show")
def service_logs(follow: bool, lines: int):
    """Show service logs from journalctl."""
    cmd = ["journalctl", "--user", "-u", "okb-db", "-u", "okb", "--no-pager", "-n", str(lines)]
    if follow:
        cmd.append("-f")

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


def entry_point():
    """Entry point that catches InsecureConfigError for clean exit."""
    try:
        main()
    except InsecureConfigError:
        # Warning already printed to stderr by check_file_permissions
        sys.exit(1)


if __name__ == "__main__":
    entry_point()
