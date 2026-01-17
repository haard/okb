"""Plugin system for LKB - extensible file parsers and API sources."""

# Re-export Document from ingest for plugin authors
from ..ingest import Document
from .base import APISource, FileParser, SyncState
from .registry import PluginRegistry

__all__ = ["FileParser", "APISource", "SyncState", "Document", "PluginRegistry"]
