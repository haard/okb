"""Dropbox Paper API source for syncing Paper documents as markdown."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from okb.ingest import Document
    from okb.plugins.base import SyncState


class DropboxPaperSource:
    """API source for Dropbox Paper documents.

    Syncs Paper documents as markdown for searchable knowledge base entries.

    Config example:
        plugins:
          sources:
            dropbox-paper:
              enabled: true
              token: ${DROPBOX_TOKEN}
              folders: [/]  # Optional: filter to specific folder paths

    Usage:
        lkb sync run dropbox-paper
        lkb sync run dropbox-paper --full  # Ignore incremental state
    """

    name = "dropbox-paper"
    source_type = "dropbox-paper"

    def __init__(self) -> None:
        self._client = None
        self._folders: list[str] | None = None
        self._doc_ids: list[str] | None = None

    def configure(self, config: dict) -> None:
        """Initialize Dropbox client with OAuth token.

        Args:
            config: Source configuration containing 'token' and optional 'folders' or 'doc_ids'
        """
        import dropbox

        token = config.get("token")
        if not token:
            raise ValueError("dropbox-paper source requires 'token' in config")

        self._client = dropbox.Dropbox(token)
        self._folders = config.get("folders")
        self._doc_ids = config.get("doc_ids")  # Specific doc IDs from CLI

    def fetch(self, state: SyncState | None = None) -> tuple[list[Document], SyncState]:
        """Fetch Paper documents from Dropbox.

        Uses the legacy Paper API to list and download documents as markdown.
        Supports incremental sync via cursor-based pagination.

        Args:
            state: Previous sync state for incremental updates, or None for full sync

        Returns:
            Tuple of (list of documents, new sync state)
        """
        from okb.plugins.base import SyncState as SyncStateClass

        if self._client is None:
            raise RuntimeError("Source not configured. Call configure() first.")

        documents: list[Document] = []
        cursor = state.cursor if state else None

        print("Fetching Dropbox Paper documents...", file=sys.stderr)

        # Use specific doc IDs from CLI, or list all Paper docs
        if self._doc_ids:
            doc_ids = self._doc_ids
            print(f"Syncing {len(doc_ids)} specific document(s)", file=sys.stderr)
        else:
            doc_ids = self._list_paper_docs(cursor)
            print(f"Found {len(doc_ids)} Paper documents", file=sys.stderr)

        for doc_id in doc_ids:
            try:
                doc = self._fetch_paper_doc(doc_id)
                if doc:
                    # Apply folder filter if configured
                    if self._folders:
                        folder_path = doc.metadata.extra.get("folder_path", "/")
                        if not any(folder_path.startswith(f) for f in self._folders):
                            continue
                    documents.append(doc)
                    print(f"  Synced: {doc.title}", file=sys.stderr)
            except Exception as e:
                print(f"  Error fetching doc {doc_id}: {e}", file=sys.stderr)

        # Build new sync state
        new_state = SyncStateClass(
            last_sync=datetime.now(UTC),
            cursor=cursor,  # Paper API doesn't provide incremental cursors
        )

        return documents, new_state

    def _list_paper_docs(self, cursor: str | None = None) -> list[str]:
        """List all Paper document IDs.

        Args:
            cursor: Pagination cursor (not used by Paper API list)

        Returns:
            List of Paper document IDs
        """

        doc_ids = []

        # Initial request
        result = self._client.paper_docs_list()
        doc_ids.extend(result.doc_ids)

        # Paginate through all results
        while result.has_more:
            result = self._client.paper_docs_list_continue(result.cursor.value)
            doc_ids.extend(result.doc_ids)

        return doc_ids

    def _fetch_paper_doc(self, doc_id: str) -> Document | None:
        """Fetch a single Paper document and convert to Document.

        Args:
            doc_id: Dropbox Paper document ID

        Returns:
            Document instance or None if fetch failed
        """
        from dropbox.paper import ExportFormat

        from okb.ingest import Document, DocumentMetadata

        # Get document metadata
        try:
            folder_result = self._client.paper_docs_get_folder_info(doc_id)
            folder_path = folder_result.folder_sharing_policy_type.name if folder_result else "/"
            # Try to get actual folder path from folders list
            if folder_result and hasattr(folder_result, "folders") and folder_result.folders:
                folder_path = "/" + "/".join(f.name for f in folder_result.folders)
            else:
                folder_path = "/"
        except Exception:
            folder_path = "/"

        # Download as markdown
        result, response = self._client.paper_docs_download(
            doc_id, ExportFormat.markdown
        )

        content = response.content.decode("utf-8")
        if not content.strip():
            return None

        # Extract title from first heading or filename
        title = result.title or f"Paper Doc {doc_id}"

        # Parse modification time
        doc_date = None
        if hasattr(result, "server_modified"):
            doc_date = result.server_modified.isoformat()

        metadata = DocumentMetadata(
            extra={
                "folder_path": folder_path,
                "doc_id": doc_id,
            }
        )
        if doc_date:
            metadata.extra["document_date"] = doc_date

        return Document(
            source_path=f"dropbox://paper/{doc_id}",
            source_type=self.source_type,
            title=title,
            content=content,
            metadata=metadata,
        )
