"""
Rescan indexed documents for freshness and re-ingest changed files.

Checks stored file_modified_at metadata against actual file mtime,
and re-ingests documents that have changed or been deleted.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


@dataclass
class RescanResult:
    """Result of a rescan operation."""

    updated: list[str] = field(default_factory=list)  # Re-ingested files
    deleted: list[str] = field(default_factory=list)  # Removed (--delete flag)
    missing: list[str] = field(default_factory=list)  # No longer exist (without --delete)
    unchanged: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)  # (path, error_message)


# Virtual path prefixes that should be skipped (handled by sync or manual re-fetch)
VIRTUAL_PREFIXES = ("claude://", "todoist://", "dropbox://", "http://", "https://")

# Source types that are file-based
FILE_SOURCE_TYPES = ("markdown", "code", "text", "org", "org-todo", "pdf", "docx")


def _escape_like(text: str) -> str:
    """Escape special LIKE pattern characters (%, _, \\) so they match literally."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Rescanner:
    """Checks indexed documents for freshness and re-ingests changed files."""

    def __init__(self, db_url: str, use_modal: bool = True):
        self.db_url = db_url
        self.use_modal = use_modal

    def get_indexed_files(self) -> list[dict]:
        """
        Query file-based documents with stored mtime.

        Returns list of dicts with:
        - base_path: The file path (without :: anchor)
        - source_type: Document source type
        - stored_mtime: ISO timestamp from metadata
        - derived_count: Number of derived documents (e.g., org-todo items)
        """
        with psycopg.connect(self.db_url, row_factory=dict_row) as conn:
            # Get unique file paths with their mtimes
            # Use SPLIT_PART to get base path (before ::)
            # Group by base path to count derived documents
            results = conn.execute(
                """
                SELECT
                    SPLIT_PART(source_path, '::', 1) as base_path,
                    MAX(source_type) as source_type,
                    MAX(metadata->>'file_modified_at') as stored_mtime,
                    COUNT(*) - 1 as derived_count
                FROM documents
                WHERE source_type = ANY(%s)
                GROUP BY SPLIT_PART(source_path, '::', 1)
                ORDER BY base_path
                """,
                (list(FILE_SOURCE_TYPES),),
            ).fetchall()

            # Filter out virtual paths
            return [
                dict(r)
                for r in results
                if not any(r["base_path"].startswith(p) for p in VIRTUAL_PREFIXES)
            ]

    def check_file_freshness(self, doc: dict) -> tuple[str, str | None]:
        """
        Check if a file is fresh, stale, or missing.

        Args:
            doc: Dict with base_path and stored_mtime

        Returns:
            Tuple of (status, current_mtime_iso) where status is:
            - 'fresh': File unchanged
            - 'stale': File modified since indexing
            - 'missing': File no longer exists
        """
        path = Path(doc["base_path"])

        if not path.exists():
            return ("missing", None)

        # Get current mtime
        current_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        current_mtime_iso = current_mtime.isoformat()

        stored_mtime = doc.get("stored_mtime")
        if not stored_mtime:
            # No stored mtime - treat as stale
            return ("stale", current_mtime_iso)

        # Parse stored mtime for comparison
        # Handle both timezone-aware and naive ISO formats
        try:
            stored_dt = datetime.fromisoformat(stored_mtime.replace("Z", "+00:00"))
            if stored_dt.tzinfo is None:
                stored_dt = stored_dt.replace(tzinfo=UTC)
        except (ValueError, AttributeError):
            return ("stale", current_mtime_iso)

        # Compare timestamps (allow 1 second tolerance for filesystem precision)
        if abs((current_mtime - stored_dt).total_seconds()) <= 1:
            return ("fresh", current_mtime_iso)

        return ("stale", current_mtime_iso)

    def delete_document_and_derived(self, source_path: str) -> int:
        """
        Delete a document and any derived documents (e.g., org-todo items).

        Args:
            source_path: The base file path

        Returns:
            Number of documents deleted
        """
        with psycopg.connect(self.db_url) as conn:
            # Delete exact match and any derived docs (path::*)
            # Escape LIKE metacharacters (%, _) so they match literally
            escaped_path = _escape_like(source_path)
            result = conn.execute(
                """
                DELETE FROM documents
                WHERE source_path = %s OR source_path LIKE %s ESCAPE '\\'
                RETURNING id
                """,
                (source_path, escaped_path + "::%"),
            ).fetchall()
            conn.commit()
            return len(result)

    def rescan(
        self,
        dry_run: bool = False,
        delete_missing: bool = False,
        verbose: bool = True,
    ) -> RescanResult:
        """
        Scan indexed files for freshness and re-ingest changed ones.

        Args:
            dry_run: If True, only report what would be done
            delete_missing: If True, remove documents for missing files
            verbose: If True, print progress to stderr

        Returns:
            RescanResult with lists of updated/deleted/missing files
        """
        from .ingest import Ingester, parse_document

        result = RescanResult()
        indexed_files = self.get_indexed_files()

        if verbose:
            print(f"Found {len(indexed_files)} indexed files", file=sys.stderr)

        ingester = None if dry_run else Ingester(self.db_url, use_modal=self.use_modal)

        for doc in indexed_files:
            base_path = doc["base_path"]
            derived_count = doc.get("derived_count", 0)
            status, current_mtime = self.check_file_freshness(doc)

            if status == "fresh":
                result.unchanged += 1
                continue

            elif status == "missing":
                derived_note = f" (+ {derived_count} derived)" if derived_count else ""
                if delete_missing:
                    if verbose:
                        print(f"  [DELETE]  {base_path}{derived_note}", file=sys.stderr)
                    if not dry_run:
                        self.delete_document_and_derived(base_path)
                    result.deleted.append(base_path)
                else:
                    if verbose:
                        print(f"  [MISSING] {base_path}{derived_note}", file=sys.stderr)
                    result.missing.append(base_path)

            elif status == "stale":
                derived_note = f" (+ {derived_count} derived)" if derived_count else ""
                if verbose:
                    print(f"  [STALE]   {base_path}{derived_note}", file=sys.stderr)

                if not dry_run:
                    try:
                        # Delete old document(s) first
                        self.delete_document_and_derived(base_path)

                        # Re-ingest
                        path = Path(base_path)
                        documents = parse_document(path)
                        if documents:
                            # Set file_modified_at on all documents
                            for d in documents:
                                d.metadata.extra["file_modified_at"] = current_mtime
                            ingester.ingest_documents(documents)
                        result.updated.append(base_path)
                    except Exception as e:
                        result.errors.append((base_path, str(e)))
                        if verbose:
                            print(f"    ERROR: {e}", file=sys.stderr)
                else:
                    result.updated.append(base_path)

        return result
