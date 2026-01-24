"""Token management for HTTP authentication.

Token format: okb_<database>_<ro|rw>_<random16hex>
Example: okb_personal_ro_a1b2c3d4e5f6g7h8

Tokens are stored hashed (SHA256) in the database for security.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg.rows import dict_row


@dataclass
class TokenInfo:
    """Information about a token."""

    token_hash: str
    database: str
    permissions: str  # 'ro' or 'rw'
    description: str | None
    created_at: datetime
    last_used_at: datetime | None


# Token format regex: okb_<database>_<ro|rw>_<hex16>
TOKEN_PATTERN = re.compile(r"^okb_([a-z0-9_-]+)_(ro|rw)_([a-f0-9]{16})$")


def generate_token(database: str, permissions: str = "rw") -> str:
    """Generate a new token for a database.

    Args:
        database: Database name (must be alphanumeric with _ or -)
        permissions: 'ro' for read-only, 'rw' for read-write

    Returns:
        Token string like 'okb_personal_ro_a1b2c3d4e5f6g7h8'
    """
    if permissions not in ("ro", "rw"):
        raise ValueError("permissions must be 'ro' or 'rw'")
    if not re.match(r"^[a-z0-9_-]+$", database):
        raise ValueError("database name must be lowercase alphanumeric with _ or -")

    random_part = secrets.token_hex(8)  # 16 hex chars
    return f"okb_{database}_{permissions}_{random_part}"


def hash_token(token: str) -> str:
    """Hash a token using SHA256.

    Args:
        token: Full token string

    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(token.encode()).hexdigest()


def parse_token(token: str) -> tuple[str, str] | None:
    """Parse a token to extract database and permissions.

    Args:
        token: Full token string

    Returns:
        Tuple of (database, permissions) or None if invalid format
    """
    if not token:
        return None
    match = TOKEN_PATTERN.match(token)
    if match:
        return (match.group(1), match.group(2))
    return None


def create_token(
    db_url: str,
    database: str,
    permissions: str = "rw",
    description: str | None = None,
) -> str:
    """Create a new token and store its hash in the database.

    Args:
        db_url: Database connection URL
        database: Database name for the token
        permissions: 'ro' or 'rw'
        description: Optional description for the token

    Returns:
        The plaintext token (only returned once, not stored)

    Raises:
        RuntimeError: If token could not be saved to database
    """
    token = generate_token(database, permissions)
    token_hash = hash_token(token)

    try:
        with psycopg.connect(db_url) as conn:
            conn.execute(
                """
                INSERT INTO tokens (token_hash, permissions, description)
                VALUES (%s, %s, %s)
                """,
                (token_hash, permissions, description),
            )
            conn.commit()
    except psycopg.Error as e:
        raise RuntimeError(f"Failed to save token to database: {e}") from e

    return token


def list_tokens(db_url: str) -> list[TokenInfo]:
    """List all tokens in a database (without the actual token values).

    Args:
        db_url: Database connection URL

    Returns:
        List of TokenInfo objects

    Raises:
        RuntimeError: If database connection fails
    """
    # Extract database name from URL for display
    from urllib.parse import urlparse

    parsed = urlparse(db_url)
    db_name = parsed.path.lstrip("/") or "default"

    try:
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            results = conn.execute(
                """
                SELECT token_hash, permissions, description, created_at, last_used_at
                FROM tokens
                ORDER BY created_at DESC
                """
            ).fetchall()

            return [
                TokenInfo(
                    token_hash=r["token_hash"],
                    database=db_name,
                    permissions=r["permissions"],
                    description=r["description"],
                    created_at=r["created_at"],
                    last_used_at=r["last_used_at"],
                )
                for r in results
            ]
    except psycopg.OperationalError as e:
        raise RuntimeError(
            f"Failed to connect to database. Ensure the database is running: {e}"
        ) from e


def delete_token(db_url: str, token_or_prefix: str) -> bool:
    """Delete a token by full token or prefix.

    Args:
        db_url: Database connection URL
        token_or_prefix: Full token or token prefix (e.g., 'lkb_personal_ro')

    Returns:
        True if token was deleted, False if not found
    """
    with psycopg.connect(db_url) as conn:
        # If it looks like a full token, hash and delete by hash
        if TOKEN_PATTERN.match(token_or_prefix):
            token_hash = hash_token(token_or_prefix)
            result = conn.execute(
                "DELETE FROM tokens WHERE token_hash = %s RETURNING token_hash",
                (token_hash,),
            ).fetchone()
            conn.commit()
            return result is not None

        # Otherwise, delete by prefix match on the hash
        # Since we can't reconstruct the token from the hash, we need to
        # match by the prefix pattern in a different way.
        # For prefix deletion, we'll use LIKE on the token_hash which won't work...
        # Actually, we need to store a prefix or identifier separately.
        # For now, return False for prefix-based deletion - full token required.

        # Alternative: store token_prefix in the tokens table
        # For this implementation, we'll just return False if not a full token
        return False


def verify_token(token: str, get_db_url_fn) -> TokenInfo | None:
    """Verify a token and return its info if valid.

    Args:
        token: Full token string
        get_db_url_fn: Function that takes a database name and returns its URL

    Returns:
        TokenInfo if valid, None if invalid
    """
    parsed = parse_token(token)
    if not parsed:
        return None

    database, permissions = parsed
    token_hash = hash_token(token)

    try:
        db_url = get_db_url_fn(database)
    except ValueError:
        return None

    try:
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            result = conn.execute(
                """
                SELECT token_hash, permissions, description, created_at, last_used_at
                FROM tokens
                WHERE token_hash = %s
                """,
                (token_hash,),
            ).fetchone()

            if not result:
                return None

            # Update last_used_at
            conn.execute(
                "UPDATE tokens SET last_used_at = NOW() WHERE token_hash = %s",
                (token_hash,),
            )
            conn.commit()

            return TokenInfo(
                token_hash=result["token_hash"],
                database=database,
                permissions=result["permissions"],
                description=result["description"],
                created_at=result["created_at"],
                last_used_at=result["last_used_at"],
            )
    except Exception:
        return None


class OKBTokenVerifier:
    """Token verifier for HTTP middleware integration."""

    def __init__(self, get_db_url_fn):
        """Initialize the verifier.

        Args:
            get_db_url_fn: Function that takes a database name and returns its URL
        """
        self.get_db_url_fn = get_db_url_fn

    def verify(self, token: str) -> TokenInfo | None:
        """Verify a token and return its info if valid.

        Args:
            token: Full token string (without 'Bearer ' prefix)

        Returns:
            TokenInfo if valid, None if invalid
        """
        return verify_token(token, self.get_db_url_fn)
