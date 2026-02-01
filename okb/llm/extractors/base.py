"""Base types for document enrichment extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExtractedTodo:
    """A TODO item extracted from document content."""

    title: str
    content: str | None = None
    due_date: datetime | None = None
    priority: int | None = None  # 1-5, 1=highest
    assignee: str | None = None
    confidence: float = 1.0
    source_context: str | None = None  # Text snippet where TODO was found


@dataclass
class ExtractedEntity:
    """An entity extracted from document content."""

    name: str
    entity_type: str  # person, project, technology, concept, organization
    aliases: list[str] = field(default_factory=list)
    description: str | None = None
    mentions: list[str] = field(default_factory=list)  # Context snippets
    confidence: float = 1.0


@dataclass
class EnrichmentResult:
    """Results from document enrichment."""

    todos: list[ExtractedTodo] = field(default_factory=list)
    entities: list[ExtractedEntity] = field(default_factory=list)

    @property
    def has_extractions(self) -> bool:
        """Check if any extractions were made."""
        return bool(self.todos or self.entities)
