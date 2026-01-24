#!/usr/bin/env python3
"""
Watch directories and auto-ingest changed files.

Usage:
    python scripts/watch.py ~/notes ~/docs
    python scripts/watch.py ~/notes --local  # Use CPU embedding
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from ..config import config
from ..ingest import Ingester, parse_document, content_hash, check_file_skip, read_text_with_fallback


class KnowledgeHandler(FileSystemEventHandler):
    """Handle file system events for knowledge base updates."""

    def __init__(self, ingester: Ingester, debounce_seconds: float = 2.0):
        self.ingester = ingester
        self.debounce_seconds = debounce_seconds
        self._pending: dict[str, float] = {}  # path -> last_event_time
        self._processed_hashes: dict[str, str] = {}  # path -> content_hash

    def _should_process(self, path: Path) -> bool:
        """Check if file should be processed."""
        if not path.is_file():
            return False

        if config.should_skip_path(path):
            return False

        if path.suffix not in config.all_extensions:
            return False

        # Check block/skip patterns (e.g., .env, *.min.js, temp files)
        skip_check = check_file_skip(path)
        if skip_check.should_skip:
            return False

        return True

    def _debounced_update(self, path: Path):
        """Update document with debouncing."""
        path_str = str(path)
        now = time.time()

        # Check debounce
        if path_str in self._pending:
            if now - self._pending[path_str] < self.debounce_seconds:
                self._pending[path_str] = now
                return

        self._pending[path_str] = now

        # Check if content actually changed
        try:
            content = read_text_with_fallback(path)
            new_hash = content_hash(content)

            if self._processed_hashes.get(path_str) == new_hash:
                return  # No actual change

            self._processed_hashes[path_str] = new_hash

            # Content-based checks (secrets, minified)
            if config.scan_content:
                skip_check = check_file_skip(path, content)
                if skip_check.should_skip:
                    prefix = "BLOCKED" if skip_check.is_security else "Skipping"
                    print(f"[watch] {prefix}: {path} ({skip_check.reason})")
                    return

        except Exception:
            return

        # Process the file
        print(f"[watch] Updating: {path}")

        try:
            doc = parse_document(path)
            # Capture file mtime for staleness tracking
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            doc.metadata.extra["file_modified_at"] = mtime.isoformat()
            self.ingester.ingest_documents([doc])

        except Exception as e:
            print(f"[watch] Error processing {path}: {e}", file=sys.stderr)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if self._should_process(path):
            self._debounced_update(path)

    def on_created(self, event: FileSystemEvent):
        """Handle new file creation."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if self._should_process(path):
            self._debounced_update(path)

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        path_str = str(path.resolve())

        # Clean up tracking
        self._pending.pop(str(path), None)
        self._processed_hashes.pop(str(path), None)

        # Remove from database
        print(f"[watch] Removing: {path}")
        try:
            self.ingester.delete_document(path_str)
        except Exception as e:
            print(f"[watch] Error removing {path}: {e}", file=sys.stderr)

    def on_moved(self, event: FileSystemEvent):
        """Handle file rename/move."""
        # Treat as delete + create
        if hasattr(event, "src_path"):
            src = Path(event.src_path)
            self._pending.pop(str(src), None)
            self._processed_hashes.pop(str(src), None)

            print(f"[watch] Removing (moved): {src}")
            try:
                self.ingester.delete_document(str(src.resolve()))
            except Exception:
                pass

        if hasattr(event, "dest_path"):
            dest = Path(event.dest_path)
            if self._should_process(dest):
                self._debounced_update(dest)


def watch(directories: list[Path], db_url: str, use_modal: bool = True):
    """Watch directories for changes."""
    ingester = Ingester(db_url, use_modal=use_modal)
    handler = KnowledgeHandler(ingester)
    observer = Observer()

    for directory in directories:
        if not directory.exists():
            print(f"Warning: Directory does not exist: {directory}", file=sys.stderr)
            continue

        observer.schedule(handler, str(directory), recursive=True)
        print(f"[watch] Watching: {directory}")

    observer.start()
    print("[watch] Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[watch] Stopping...")
        observer.stop()

    observer.join()
    print("[watch] Done.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Watch directories and auto-ingest changes")
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=[Path.home() / "notes"],
        help="Directories to watch (default: ~/notes)",
    )
    parser.add_argument("--db-url", default=config.db_url, help="Database URL")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local CPU embedding instead of Modal",
    )

    args = parser.parse_args()
    watch(args.directories, args.db_url, use_modal=not args.local)


if __name__ == "__main__":
    main()
