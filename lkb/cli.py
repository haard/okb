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
@click.version_option(package_name="local-kb")
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
    result = subprocess.run(
        ["docker", "container", "inspect", "-f", "{{.State.Status}}", config.docker_container_name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _get_init_sql_path() -> Path:
    """Get the path to init.sql, extracting from package if needed."""
    # Try to access init.sql from package data
    try:
        ref = importlib.resources.files("lkb.data").joinpath("init.sql")
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
        )
        if result.returncode == 0:
            click.echo(" ready.")
            return True
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
        f"postgresql://knowledge:{config.docker_password}@"
        f"localhost:{config.docker_port}/postgres"
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
        result = subprocess.run(
            ["docker", "start", config.docker_container_name],
            capture_output=True,
            text=True,
        )
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
        ref = importlib.resources.files("lkb.data").joinpath("init.sql")
        init_sql_content = ref.read_text()
        # Write to temp file
        temp_dir = Path(tempfile.gettempdir()) / "lkb"
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

    result = subprocess.run(cmd, capture_output=True, text=True)
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
    result = subprocess.run(
        ["docker", "stop", config.docker_container_name],
        capture_output=True,
        text=True,
    )
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
        click.echo("Run 'lkb db start' to create it.")
        return

    click.echo(f"Container: {config.docker_container_name}")
    click.echo(f"Status: {container_status}")
    click.echo(f"Port: {config.docker_port}")
    click.echo(f"Volume: {config.docker_volume_name}")

    if container_status == "running":
        # Try to get more info
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
        )
        if result.returncode == 0:
            click.echo("Database: ready")
            # Show migration status
            try:
                from .migrate import get_applied, get_pending

                applied = get_applied(config.db_url)
                pending = get_pending(config.db_url)
                click.echo(f"Migrations: {len(applied)} applied, {len(pending)} pending")
                if pending:
                    click.echo("  Run 'lkb db migrate' to apply pending migrations.")
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
    """
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
    )
    click.echo(f"Removed container '{config.docker_container_name}'.")

    # Remove volume
    subprocess.run(
        ["docker", "volume", "rm", config.docker_volume_name],
        capture_output=True,
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
    """Create default config file at ~/.config/lkb/config.yaml."""
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
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--metadata", "-m", default="{}", help="JSON metadata to attach")
@click.option("--local", is_flag=True, help="Use local CPU embedding instead of Modal")
@click.option("--db", "database", default=None, help="Database to ingest into")
@click.pass_context
def ingest(ctx, paths: tuple[str, ...], metadata: str, local: bool, database: str | None):
    """Ingest documents into the knowledge base."""
    import json as json_module
    from pathlib import Path

    from .ingest import Ingester, collect_documents, parse_code, parse_markdown

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
        path = Path(path_str).resolve()
        if path.is_dir():
            documents.extend(collect_documents(path, extra_metadata))
        elif path.is_file():
            if path.suffix in config.document_extensions:
                documents.append(parse_markdown(path, extra_metadata))
            elif path.suffix in config.code_extensions:
                documents.append(parse_code(path, extra_metadata))
            else:
                click.echo(f"Skipping unsupported file: {path}", err=True)

    if not documents:
        click.echo("No documents found to ingest.")
        return

    click.echo(f"Found {len(documents)} documents to process")
    ingester.ingest_documents(documents)
    click.echo("Done!")


# =============================================================================
# Serve command
# =============================================================================


@main.command()
@click.option("--db", "database", default=None, help="Database to serve")
@click.pass_context
def serve(ctx, database: str | None):
    """Start the MCP server for Claude Code integration."""
    import asyncio

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
    sys.argv = ["lkb-watch"] + list(paths)
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


if __name__ == "__main__":
    main()
