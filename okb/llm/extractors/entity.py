"""Entity extraction from document content using LLM."""

from __future__ import annotations

import json
import re

from .base import ExtractedEntity

ENTITY_SYSTEM_PROMPT = """\
You are an expert at identifying named entities in text for a PERSONAL knowledge base.
Extract only entities that are specific to the author's context - things an LLM wouldn't know about.

Entity types to extract:
- person: People the author knows, works with, or references (colleagues, contacts, clients)
- project: Specific named projects/products/codebases (e.g., "Acme Dashboard", "customer-portal")
          NOT git branches, environments, or workflow stages
- technology: ONLY obscure/niche tools or internal systems - NOT well-known technologies
- organization: Specific companies, teams, clients the author works with

DO NOT extract:
- Well-known technologies: JSON, HTTP, SQL, Python, JavaScript, Docker, AWS, PostgreSQL, React, etc.
  (The LLM already knows these - they add no value to a personal knowledge base)
- Code symbols: function names, method calls, variables, class names
- Generic terms: "user", "data", "system", "database", "API", "server", "client"
- Git branches/workflow terms: main, master, develop, release, staging, production, feature, hotfix
- Generic process terms: deploy, build, test, migration, setup, config
- Environment names: dev, prod, qa, uat, local
- Issue or bug descriptions - those are documents, not entities
- Famous people, major companies (Google, Microsoft, etc.) unless contextually relevant to author

ONLY extract entities that would help answer "Who/what is X?" where X is specific to this person.

For each entity found, extract:
- name: The canonical name (proper noun)
- entity_type: One of: person, project, technology, organization
- aliases: Other names/abbreviations (optional)
- description: Brief description based on context (optional)
- mentions: Text snippets where entity appears (max 3)
- confidence: How confident you are (0.0-1.0)

Return JSON array. Return empty array [] if no context-specific entities found.
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

    return _parse_entity_response(response.content, min_confidence, title)


def _looks_like_code(name: str) -> bool:
    """Check if entity name looks like code."""
    # Contains parentheses (function calls)
    if "(" in name or ")" in name:
        return True
    # Snake_case with underscores (likely variable/function)
    if "_" in name and name.islower():
        return True
    # Starts with lowercase and contains dots (method chain)
    if name and name[0].islower() and "." in name:
        return True
    # CamelCase starting with lowercase (variable/method name)
    if name and name[0].islower() and any(c.isupper() for c in name):
        return True
    return False


# Well-known technologies that add no value to a personal knowledge base
COMMON_TECHNOLOGIES = frozenset(
    s.lower()
    for s in [
        # Data formats
        "JSON",
        "XML",
        "YAML",
        "CSV",
        "HTML",
        "CSS",
        "Markdown",
        # Protocols
        "HTTP",
        "HTTPS",
        "REST",
        "GraphQL",
        "WebSocket",
        "TCP",
        "UDP",
        "SSH",
        "FTP",
        "SMTP",
        # Languages
        "Python",
        "JavaScript",
        "TypeScript",
        "Java",
        "Go",
        "Rust",
        "C",
        "C++",
        "Ruby",
        "PHP",
        "Swift",
        "Kotlin",
        "Scala",
        "Bash",
        "Shell",
        "SQL",
        "Lua",
        # Major frameworks/tools
        "React",
        "Vue",
        "Angular",
        "Node.js",
        "Django",
        "Flask",
        "FastAPI",
        "Rails",
        "Spring",
        "Express",
        "Next.js",
        # Databases
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "SQLite",
        "Elasticsearch",
        "DynamoDB",
        # Cloud/infra
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "Linux",
        "Windows",
        "macOS",
        "Nginx",
        "Apache",
        # Tools
        "Git",
        "GitHub",
        "GitLab",
        "npm",
        "pip",
        "Webpack",
        "VS Code",
        "Vim",
        "Emacs",
    ]
)

# Generic git/workflow/environment terms that are not context-specific
GENERIC_TERMS = frozenset(
    s.lower()
    for s in [
        # Git branches
        "main",
        "master",
        "develop",
        "development",
        "release",
        "staging",
        "production",
        "feature",
        "hotfix",
        "bugfix",
        # Environments
        "dev",
        "prod",
        "test",
        "qa",
        "uat",
        "local",
        "sandbox",
        # Workflow/process terms
        "deploy",
        "build",
        "migration",
        "setup",
        "config",
        "configuration",
        "rollback",
        "rollout",
        # Generic architectural terms
        "frontend",
        "backend",
        "api",
        "service",
        "server",
        "client",
        "app",
        "application",
        "module",
        "component",
        "library",
        "package",
        "plugin",
        "extension",
        # Generic data terms
        "database",
        "cache",
        "queue",
        "worker",
        "scheduler",
        "cron",
    ]
)


def _parse_entity_response(
    response_text: str, min_confidence: float, title: str = ""
) -> list[ExtractedEntity]:
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

    valid_types = {"person", "project", "technology", "organization"}
    entities = []

    for item in entities_data:
        if not isinstance(item, dict):
            continue

        name = item.get("name", "").strip()
        entity_type = item.get("entity_type")

        if not name or not isinstance(name, str):
            continue
        if not entity_type or entity_type not in valid_types:
            continue

        # Filter: too short or too long
        if len(name) < 3 or len(name) > 80:
            continue

        # Filter: looks like code
        if _looks_like_code(name):
            continue

        # Filter: well-known technologies (LLM already knows these)
        if name.lower() in COMMON_TECHNOLOGIES:
            continue

        # Filter: generic git/workflow/environment terms
        if name.lower() in GENERIC_TERMS:
            continue

        # Filter: matches document title (source shouldn't be extracted as entity)
        if title and name.lower() == title.lower():
            continue

        # Get confidence (default to 0.85 if not specified)
        confidence = item.get("confidence", 0.85)
        if not isinstance(confidence, int | float):
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
                name=name,
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
