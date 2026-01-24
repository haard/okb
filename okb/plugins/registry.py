"""Plugin discovery and registration via entry_points."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import APISource, FileParser


class PluginRegistry:
    """Registry for file parsers and API sources discovered via entry_points.

    Plugins are discovered from two entry_point groups:
    - okb.parsers: FileParser implementations
    - okb.sources: APISource implementations

    Example pyproject.toml for a plugin:
        [project.entry-points."okb.parsers"]
        epub = "okb_epub:EpubParser"

        [project.entry-points."okb.sources"]
        github = "okb_github:GitHubSource"
    """

    _parsers: dict[str, list[FileParser]] = {}  # ext -> list of parsers
    _sources: dict[str, APISource] = {}  # name -> source
    _loaded = False

    @classmethod
    def load_plugins(cls) -> None:
        """Load all plugins from entry_points. Called automatically on first use."""
        if cls._loaded:
            return

        # Load file parsers
        parser_eps = entry_points(group="okb.parsers")
        for ep in parser_eps:
            try:
                parser_cls = ep.load()
                parser = parser_cls()
                for ext in parser.extensions:
                    ext_lower = ext.lower()
                    if ext_lower not in cls._parsers:
                        cls._parsers[ext_lower] = []
                    cls._parsers[ext_lower].append(parser)
            except Exception as e:
                print(f"Warning: Failed to load parser plugin '{ep.name}': {e}")

        # Load API sources
        source_eps = entry_points(group="okb.sources")
        for ep in source_eps:
            try:
                source_cls = ep.load()
                source = source_cls()
                cls._sources[source.name] = source
            except Exception as e:
                print(f"Warning: Failed to load source plugin '{ep.name}': {e}")

        cls._loaded = True

    @classmethod
    def get_parser_for_file(cls, path: Path) -> FileParser | None:
        """Find a parser that can handle this file.

        First filters by extension, then calls can_parse() on each candidate.

        Args:
            path: Path to the file to parse

        Returns:
            FileParser instance that can handle the file, or None
        """
        cls.load_plugins()
        ext = path.suffix.lower()
        for parser in cls._parsers.get(ext, []):
            if parser.can_parse(path):
                return parser
        return None

    @classmethod
    def get_source(cls, name: str) -> APISource | None:
        """Get an API source by name.

        Args:
            name: Source name (e.g., 'github', 'todoist')

        Returns:
            APISource instance, or None if not found
        """
        cls.load_plugins()
        return cls._sources.get(name)

    @classmethod
    def list_sources(cls) -> list[str]:
        """List all available API source names.

        Returns:
            List of source names
        """
        cls.load_plugins()
        return list(cls._sources.keys())

    @classmethod
    def list_parsers(cls) -> dict[str, list[str]]:
        """List all registered parsers by extension.

        Returns:
            Dict mapping extension to list of parser source_type names
        """
        cls.load_plugins()
        return {
            ext: [p.source_type for p in parsers] for ext, parsers in cls._parsers.items()
        }

    @classmethod
    def reset(cls) -> None:
        """Reset the registry. Mainly useful for testing."""
        cls._parsers = {}
        cls._sources = {}
        cls._loaded = False
