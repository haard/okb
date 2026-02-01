"""Entity extraction from document content using LLM."""

from __future__ import annotations

import json
import re

from .base import ExtractedEntity

ENTITY_SYSTEM_PROMPT = """\
You are an expert at identifying named entities in text.
Extract entities from the given document content.

Entity types to extract:
- person: Named individuals (colleagues, contacts, authors, historical figures)
- project: Project names, codebases, products, initiatives
- technology: Tools, frameworks, languages, services, platforms
- concept: Technical concepts, methodologies, patterns, theories
- organization: Companies, teams, groups, institutions

For each entity found, extract:
- name: The canonical name of the entity
- entity_type: One of: person, project, technology, concept, organization
- aliases: Other names/abbreviations for this entity (optional)
- description: Brief description based on context (optional)
- mentions: List of text snippets where entity appears (max 3)
- confidence: How confident you are (0.0-1.0)

Guidelines:
- Extract entities that are significant to understanding the document
- Don't extract generic terms (e.g., "user", "data", "system")
- Don't extract common words or phrases unless they're named entities
- Prefer the full formal name as the primary name
- Include common abbreviations as aliases (e.g., "AWS" for "Amazon Web Services")

Return JSON array of entities. Return empty array [] if none found.
"""

ENTITY_USER_PROMPT = """\
Document title: {title}
Source type: {source_type}

Content:
{content}

Extract named entities as JSON array.
"""


def extract_entities(
    content: str,
    title: str,
    source_type: str | None = None,
    min_confidence: float = 0.8,
) -> list[ExtractedEntity]:
    """Extract entities from document content using LLM.

    Args:
        content: Document content to analyze
        title: Document title for context
        source_type: Type of document (optional)
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        List of extracted entities
    """
    from .. import complete

    # Truncate content if too long
    if len(content) > 20000:
        content = content[:20000] + "\n\n[... content truncated ...]"

    prompt = ENTITY_USER_PROMPT.format(
        title=title,
        source_type=source_type or "unknown",
        content=content,
    )

    response = complete(
        prompt=prompt,
        system=ENTITY_SYSTEM_PROMPT,
        max_tokens=2048,
        use_cache=True,
    )

    if response is None:
        return []

    return _parse_entity_response(response.content, min_confidence)


def _parse_entity_response(response_text: str, min_confidence: float) -> list[ExtractedEntity]:
    """Parse LLM response into ExtractedEntity objects."""
    # Try to extract JSON from response
    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if not json_match:
        return []

    try:
        entities_data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(entities_data, list):
        return []

    valid_types = {"person", "project", "technology", "concept", "organization"}
    entities = []

    for item in entities_data:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        entity_type = item.get("entity_type")

        if not name or not isinstance(name, str):
            continue
        if not entity_type or entity_type not in valid_types:
            continue

        # Get confidence (default to 0.85 if not specified)
        confidence = item.get("confidence", 0.85)
        if not isinstance(confidence, (int, float)):
            confidence = 0.85

        if confidence < min_confidence:
            continue

        # Parse aliases
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        aliases = [a for a in aliases if isinstance(a, str)]

        # Parse mentions
        mentions = item.get("mentions", [])
        if not isinstance(mentions, list):
            mentions = []
        mentions = [m for m in mentions if isinstance(m, str)][:3]

        entities.append(
            ExtractedEntity(
                name=name.strip(),
                entity_type=entity_type,
                aliases=aliases,
                description=item.get("description"),
                mentions=mentions,
                confidence=float(confidence),
            )
        )

    return entities


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for deduplication and URL generation.

    Examples:
        "John Smith" -> "john-smith"
        "AWS (Amazon Web Services)" -> "aws-amazon-web-services"
        "React.js" -> "react-js"
    """
    # Lowercase
    normalized = name.lower()
    # Replace non-alphanumeric with spaces
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    # Collapse whitespace and replace with hyphens
    normalized = re.sub(r"\s+", "-", normalized.strip())
    # Remove leading/trailing hyphens
    normalized = normalized.strip("-")
    return normalized


def entity_source_path(entity_type: str, name: str) -> str:
    """Generate source_path for an entity document.

    Format: okb://entity/{type}/{normalized-name}
    """
    normalized = normalize_entity_name(name)
    return f"okb://entity/{entity_type}/{normalized}"
