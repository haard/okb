"""Migration runner for okb database schema."""

from __future__ import annotations

from pathlib import Path

from yoyo import get_backend, read_migrations


def get_migrations_path() -> str:
    """Get path to migrations directory."""
    return str(Path(__file__).parent / "migrations")


def _convert_db_url(db_url: str) -> str:
    """Convert psycopg3 URL to yoyo-compatible format.

    yoyo uses psycopg2 by default. We convert:
    postgresql://... -> postgresql+psycopg://...
    to use psycopg v3.
    """
    if db_url.startswith("postgresql://") and "+psycopg" not in db_url:
        return db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return db_url


def run_migrations(db_url: str) -> list[str]:
    """Apply pending migrations, return list of applied migration IDs."""
    backend = get_backend(_convert_db_url(db_url))
    migrations = read_migrations(get_migrations_path())

    with backend.lock():
        to_apply = backend.to_apply(migrations)
        if to_apply:
            backend.apply_migrations(to_apply)

    return [m.id for m in to_apply]


def get_pending(db_url: str) -> list[str]:
    """Get list of pending migration IDs."""
    backend = get_backend(_convert_db_url(db_url))
    migrations = read_migrations(get_migrations_path())
    return [m.id for m in backend.to_apply(migrations)]


def get_applied(db_url: str) -> list[str]:
    """Get list of applied migration IDs."""
    backend = get_backend(_convert_db_url(db_url))
    migrations = read_migrations(get_migrations_path())
    to_apply = backend.to_apply(migrations)
    to_apply_ids = {m.id for m in to_apply}
    return [m.id for m in migrations if m.id not in to_apply_ids]
