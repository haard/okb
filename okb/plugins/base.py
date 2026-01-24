"""Protocol definitions for LKB plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..ingest import Document


@dataclass
class SyncState:
    """Tracks sync progress for incremental updates."""

    last_sync: datetime | None = None
    cursor: str | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class FileParser(Protocol):
    """Protocol for file format parsers.

    Plugins implement this to add support for new file types.

    Example:
        class MyParser:
            extensions = ['.xyz']
            source_type = 'xyz'

            def can_parse(self, path: Path) -> bool:
                return path.suffix.lower() == '.xyz'

            def parse(self, path: Path, extra_metadata: dict | None = None) -> Document:
                ...
    """

    extensions: list[str]  # e.g., ['.pdf', '.PDF'] - for fast pre-filtering
    source_type: str  # e.g., 'pdf'

    def can_parse(self, path: Path) -> bool:
        """Check if this parser can handle the file (beyond just extension).

        Called after extension match. Can inspect file content, magic bytes, etc.
        Return False to let other parsers try.
        """
        ...

    def parse(self, path: Path, extra_metadata: dict | None = None) -> Document:
        """Parse the file and return a Document.

        Args:
            path: Path to the file to parse
            extra_metadata: Optional metadata to merge into the document

        Returns:
            Document instance ready for ingestion
        """
        ...


@runtime_checkable
class APISource(Protocol):
    """Protocol for API-based data sources.

    Plugins implement this to sync data from external services.

    Example:
        class GitHubSource:
            name = 'github'
            source_type = 'github-source'

            def configure(self, config: dict) -> None:
                self._token = config['token']
                self._repos = config.get('repos', [])  # From CLI --repo flags

            def fetch(self, state: SyncState | None = None) -> tuple[list[Document], SyncState]:
                # Use state.last_sync for incremental fetching
                # Return (documents, new_state)
                ...
    """

    name: str  # e.g., 'github'
    source_type: str  # e.g., 'github-issue'

    def configure(self, config: dict) -> None:
        """Configure the source with settings from config file.

        Config values may include resolved environment variables.

        Args:
            config: Source-specific configuration dict
        """
        ...

    def fetch(self, state: SyncState | None = None) -> tuple[list[Document], SyncState]:
        """Fetch documents from the external source.

        Should support incremental fetching using the state object.

        Args:
            state: Previous sync state for incremental updates, or None for full sync

        Returns:
            Tuple of (list of documents, new sync state)
        """
        ...
