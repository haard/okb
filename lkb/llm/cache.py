"""LLM response caching to avoid redundant API calls."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING

import psycopg

from .base import LLMResponse

if TYPE_CHECKING:
    pass


def _compute_cache_key(prompt: str, system: str | None, model: str) -> str:
    """Compute cache key from prompt, system, and model.

    Args:
        prompt: User prompt
        system: System prompt (may be None)
        model: Model name

    Returns:
        SHA256 hash of the combined inputs
    """
    content = f"{prompt}\n---\n{system or ''}\n---\n{model}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached(
    prompt: str,
    system: str | None,
    provider: str,
    db_url: str | None = None,
) -> LLMResponse | None:
    """Retrieve a cached LLM response if available.

    Args:
        prompt: User prompt
        system: System prompt
        provider: Provider name (e.g., "claude")
        db_url: Database URL (default: from config)

    Returns:
        Cached LLMResponse or None if not found
    """
    from ..config import config

    if db_url is None:
        db_url = config.db_url

    # Get model from config for cache key
    model = config.llm_model or "default"
    content_hash = _compute_cache_key(prompt, system, model)

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT response FROM llm_cache
                    WHERE content_hash = %s AND provider = %s AND model = %s
                    """,
                    (content_hash, provider, model),
                )
                row = cur.fetchone()
                if row is None:
                    return None

                # Parse cached response
                data = json.loads(row[0])
                return LLMResponse(
                    content=data["content"],
                    model=data["model"],
                    input_tokens=data.get("input_tokens"),
                    output_tokens=data.get("output_tokens"),
                )
    except psycopg.Error:
        # Cache miss on error - don't block on cache failures
        return None


def cache_response(
    prompt: str,
    system: str | None,
    provider: str,
    response: LLMResponse,
    db_url: str | None = None,
) -> None:
    """Store an LLM response in the cache.

    Args:
        prompt: User prompt
        system: System prompt
        provider: Provider name
        response: LLMResponse to cache
        db_url: Database URL (default: from config)
    """
    from ..config import config

    if db_url is None:
        db_url = config.db_url

    model = config.llm_model or "default"
    content_hash = _compute_cache_key(prompt, system, model)

    # Serialize response
    data = {
        "content": response.content,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_cache (content_hash, provider, model, response)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        response = EXCLUDED.response,
                        created_at = NOW()
                    """,
                    (content_hash, provider, model, json.dumps(data)),
                )
            conn.commit()
    except psycopg.Error:
        # Don't fail on cache write errors
        pass


def clear_cache(
    older_than: datetime | None = None,
    db_url: str | None = None,
) -> int:
    """Clear cached LLM responses.

    Args:
        older_than: Only clear entries older than this datetime.
                   If None, clears all entries.
        db_url: Database URL (default: from config)

    Returns:
        Number of entries deleted
    """
    from ..config import config

    if db_url is None:
        db_url = config.db_url

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                if older_than:
                    cur.execute(
                        "DELETE FROM llm_cache WHERE created_at < %s",
                        (older_than,),
                    )
                else:
                    cur.execute("DELETE FROM llm_cache")
                deleted = cur.rowcount
            conn.commit()
            return deleted
    except psycopg.Error:
        return 0


def get_cache_stats(db_url: str | None = None) -> dict:
    """Get statistics about the LLM cache.

    Args:
        db_url: Database URL (default: from config)

    Returns:
        Dict with cache statistics
    """
    from ..config import config

    if db_url is None:
        db_url = config.db_url

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Total entries
                cur.execute("SELECT COUNT(*) FROM llm_cache")
                total = cur.fetchone()[0]

                # Entries by provider/model
                cur.execute(
                    """
                    SELECT provider, model, COUNT(*) as count
                    FROM llm_cache
                    GROUP BY provider, model
                    ORDER BY count DESC
                    """
                )
                by_provider = [
                    {"provider": r[0], "model": r[1], "count": r[2]} for r in cur.fetchall()
                ]

                # Oldest entry
                cur.execute("SELECT MIN(created_at) FROM llm_cache")
                oldest = cur.fetchone()[0]

                return {
                    "total_entries": total,
                    "by_provider": by_provider,
                    "oldest_entry": oldest.isoformat() if oldest else None,
                }
    except psycopg.Error:
        return {"total_entries": 0, "by_provider": [], "oldest_entry": None}
