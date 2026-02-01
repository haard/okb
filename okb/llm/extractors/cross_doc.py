"""Cross-document entity detection - find mentions appearing in multiple documents."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .base import ExtractedEntity


@dataclass
class CrossDocCandidate:
    """A potential entity found across multiple documents."""

    text: str  # Normalized mention text
    document_ids: list[str]
    document_count: int
    sample_contexts: list[str]  # Sample text contexts where it appears
    suggested_type: str | None = None
    confidence: float = 0.0


CLASSIFY_SYSTEM = """\
You classify text mentions as named entities. You MUST respond with ONLY valid JSON, no other text.

Entity types: person, project, technology, concept, organization, not_entity

Required JSON format:
{"classifications":[{"text":"Django","type":"technology","confidence":0.9}]}
"""

CLASSIFY_USER = """\
Classify these mentions as entities. Reply with ONLY JSON, no explanation.

{mentions}

JSON format: {{"classifications":[{{"text":"...","type":"...","confidence":0.9}}]}}
Types: person, project, technology, concept, organization, not_entity
"""


def find_cross_document_entities(
    db_url: str,
    min_documents: int = 3,
    limit: int = 100,
) -> list[CrossDocCandidate]:
    """Find text mentions appearing in multiple documents but not extracted as entities.

    Args:
        db_url: Database URL
        min_documents: Minimum documents a mention must appear in
        limit: Maximum candidates to return

    Returns:
        List of CrossDocCandidate objects
    """
    candidates: list[CrossDocCandidate] = []

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        # Get existing entity names and aliases to exclude
        existing_names = set()

        # Get entity titles
        results = conn.execute(
            "SELECT LOWER(title) as name FROM documents WHERE source_type = 'entity'"
        ).fetchall()
        existing_names.update(r["name"] for r in results)

        # Get aliases
        results = conn.execute("SELECT LOWER(alias_text) as name FROM entity_aliases").fetchall()
        existing_names.update(r["name"] for r in results)

        # Get already-detected candidates
        results = conn.execute(
            "SELECT LOWER(text) as name FROM cross_doc_entity_candidates"
        ).fetchall()
        existing_detected = {r["name"] for r in results}

        # Get documents (exclude derived documents)
        docs = conn.execute(
            """
            SELECT id, content
            FROM documents
            WHERE source_path NOT LIKE '%%::todo/%%'
            AND source_path NOT LIKE 'okb://entity/%%'
            AND source_path NOT LIKE 'claude://%%'
            AND content IS NOT NULL
            LIMIT 1000
            """
        ).fetchall()

        # Extract noun phrases and track document occurrences
        mention_docs: dict[str, set[str]] = defaultdict(set)
        mention_contexts: dict[str, list[str]] = defaultdict(list)

        for doc in docs:
            doc_id = str(doc["id"])
            content = doc["content"]
            phrases = _extract_noun_phrases(content)

            for phrase, context in phrases:
                normalized = phrase.lower().strip()
                # Skip if too short, too long, or already exists
                if len(normalized) < 2 or len(normalized) > 50:
                    continue
                if normalized in existing_names or normalized in existing_detected:
                    continue
                # Skip common words
                if normalized in COMMON_WORDS:
                    continue

                mention_docs[normalized].add(doc_id)
                if len(mention_contexts[normalized]) < 3:
                    mention_contexts[normalized].append(context[:200])

        # Filter to mentions appearing in min_documents
        for text, doc_ids in mention_docs.items():
            if len(doc_ids) >= min_documents:
                candidates.append(
                    CrossDocCandidate(
                        text=text,
                        document_ids=list(doc_ids),
                        document_count=len(doc_ids),
                        sample_contexts=mention_contexts[text],
                    )
                )

        # Sort by document count and limit
        candidates.sort(key=lambda c: c.document_count, reverse=True)
        candidates = candidates[:limit]

    return candidates


def _extract_noun_phrases(text: str) -> list[tuple[str, str]]:
    """Extract potential noun phrases from text using simple heuristics.

    Returns list of (phrase, context) tuples.
    """
    phrases = []

    # Pattern for capitalized phrases (likely proper nouns)
    # Matches: "Amazon Web Services", "John Smith", "React.js"
    cap_pattern = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\.[a-z]+)?)"

    # Pattern for tech terms with special chars
    # Matches: "C++", "Node.js", "OpenAI", "PostgreSQL"
    tech_pattern = r"\b([A-Z][a-zA-Z]+(?:\.[a-z]+|\+\+)?)\b"

    # Combined patterns
    for pattern in [cap_pattern, tech_pattern]:
        for match in re.finditer(pattern, text):
            phrase = match.group(1)
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            phrases.append((phrase, context))

    return phrases


# Common words to exclude (not named entities)
COMMON_WORDS = {
    # Articles, prepositions, conjunctions
    "the", "this", "that", "with", "from", "have", "been", "were", "will",
    "would", "could", "should", "these", "those", "then", "than", "when",
    "where", "what", "which", "while", "other", "some", "most", "many",
    "more", "each", "every", "both", "after", "before", "about", "over",
    "under", "again", "once", "here", "there", "also", "just", "only",
    "even", "still", "well", "back", "such", "very", "much", "into", "onto",
    "upon", "within", "without", "between", "among", "through", "during",
    "because", "since", "until", "unless", "although", "though", "however",
    "therefore", "thus", "hence", "meanwhile", "otherwise", "instead",
    "if", "for", "or", "and", "but", "nor", "yet", "so", "as", "at", "by",
    "to", "in", "on", "of", "up", "no", "not", "any", "all", "few",
    # Common verbs
    "make", "made", "like", "need", "want", "take", "give", "find", "keep",
    "put", "set", "get", "let", "say", "see", "use", "used", "using", "add",
    "added", "adding", "run", "running", "try", "tried", "call", "called",
    "show", "shown", "showing", "check", "checked", "checking", "include",
    "included", "including", "contain", "contains", "provide", "provides",
    "allow", "allows", "enable", "enables", "support", "supports", "handle",
    "handles", "generate", "generated", "generating", "test", "tested",
    "testing", "build", "building", "deploy", "deploying", "move", "moved",
    "send", "sent", "receive", "received", "pass", "passed", "fail", "failed",
    "complete", "completed", "finish", "finished", "done",
    # Common nouns (generic)
    "user", "users", "data", "file", "files", "code", "time", "work", "way",
    "case", "cases", "point", "points", "part", "parts", "place", "thing",
    "things", "name", "names", "number", "numbers", "type", "types", "list",
    "lists", "line", "lines", "note", "notes", "example", "examples",
    "section", "sections", "chapter", "page", "pages", "document", "documents",
    "item", "items", "entry", "entries", "record", "records", "row", "rows",
    "column", "columns", "table", "tables", "field", "fields", "form", "forms",
    "view", "views", "model", "models", "schema", "index", "key", "keys",
    "token", "tokens", "id", "ids", "url", "urls", "path", "paths",
    # Programming terms (generic)
    "function", "functions", "method", "methods", "class", "classes",
    "object", "objects", "value", "values", "result", "results", "error",
    "errors", "issue", "issues", "problem", "problems", "solution", "solutions",
    "system", "systems", "process", "processes", "service", "services",
    "server", "servers", "client", "clients", "request", "requests",
    "response", "responses", "query", "queries", "update", "updates",
    "delete", "deletes", "create", "creates", "read", "reads", "write",
    "writes", "input", "inputs", "output", "outputs", "return", "returns",
    "exception", "exceptions", "warning", "warnings", "message", "messages",
    "event", "events", "action", "actions", "task", "tasks", "job", "jobs",
    "api", "apis", "json", "xml", "html", "css", "sql", "http", "https",
    "config", "configs", "setting", "settings", "option", "options",
    "param", "params", "parameter", "parameters", "argument", "arguments",
    "variable", "variables", "constant", "constants", "property", "properties",
    "attribute", "attributes", "module", "modules", "package", "packages",
    "library", "libraries", "framework", "frameworks", "tool", "tools",
    "script", "scripts", "command", "commands", "handler", "handlers",
    "callback", "callbacks", "hook", "hooks", "plugin", "plugins",
    "context", "contexts", "state", "states", "status", "priority",
    "level", "levels", "mode", "modes", "flag", "flags", "tag", "tags",
    # Ordinals and quantifiers
    "start", "end", "first", "last", "next", "previous", "begin", "final",
    "new", "old", "good", "bad", "high", "low", "top", "bottom", "left",
    "right", "true", "false", "yes", "none", "null", "undefined", "empty",
    # Adjectives
    "default", "custom", "main", "base", "core", "common", "standard",
    "general", "specific", "local", "remote", "public", "private",
    "internal", "external", "simple", "basic", "advanced", "current",
    "available", "required", "optional", "important", "different", "same",
    "similar", "related", "following", "above", "below", "existing",
    "valid", "invalid", "active", "inactive", "enabled", "disabled",
    "visible", "hidden", "open", "closed", "full", "partial", "total",
    "rest", "remaining", "other", "another", "single", "multiple",
    # Days and months
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "today", "tomorrow", "yesterday", "week", "month", "year",
}


def classify_candidates(
    candidates: list[CrossDocCandidate],
    db_url: str | None = None,
    batch_size: int = 25,
) -> list[ExtractedEntity]:
    """Use LLM to classify cross-document candidates as entities.

    Args:
        candidates: List of candidates to classify
        db_url: Database URL (for caching)
        batch_size: Max candidates per LLM call (default 25 to avoid prompt length issues)

    Returns:
        List of ExtractedEntity objects for valid entities
    """
    if not candidates:
        return []

    from .. import complete

    all_entities = []
    candidate_map = {c.text.lower(): c for c in candidates}

    # Process in batches to avoid prompt length issues with smaller models
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]

        # Build prompt for this batch
        mention_lines = []
        for c in batch:
            mention_lines.append(f'- "{c.text}" (in {c.document_count} docs)')

        prompt = CLASSIFY_USER.format(mentions="\n".join(mention_lines))

        response = complete(prompt, system=CLASSIFY_SYSTEM, max_tokens=2048, use_cache=True)

        if response is None:
            continue

        # Parse response
        try:
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            # Fix common LLM JSON errors: leading zeros (00.9 -> 0.9), trailing commas
            content = re.sub(r":0+(\d)", r":\1", content)  # 00.9 -> 0.9
            content = re.sub(r",\s*([}\]])", r"\1", content)  # trailing commas
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        # Handle both {"classifications": [...]} and direct list formats
        if isinstance(data, list):
            classifications = data
        else:
            classifications = data.get("classifications", [])

        for cls in classifications:
            text = cls.get("text", "")
            entity_type = cls.get("type", "")
            confidence = cls.get("confidence", 0.5)

            if entity_type == "not_entity" or not entity_type:
                continue

            candidate = candidate_map.get(text.lower())
            if candidate:
                # Update candidate with classification
                candidate.suggested_type = entity_type
                candidate.confidence = confidence

                all_entities.append(
                    ExtractedEntity(
                        name=text,
                        entity_type=entity_type,
                        confidence=confidence,
                        mentions=candidate.sample_contexts[:3],
                    )
                )

    return all_entities


def store_candidates(db_url: str, candidates: list[CrossDocCandidate]) -> int:
    """Store cross-document candidates in database.

    Args:
        db_url: Database URL
        candidates: Candidates to store

    Returns:
        Number of candidates stored
    """
    stored = 0

    with psycopg.connect(db_url) as conn:
        for c in candidates:
            try:
                conn.execute(
                    """
                    INSERT INTO cross_doc_entity_candidates
                        (text, document_ids, document_count, sample_contexts,
                         suggested_type, confidence, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                    ON CONFLICT (text) DO UPDATE SET
                        document_ids = EXCLUDED.document_ids,
                        document_count = EXCLUDED.document_count,
                        sample_contexts = EXCLUDED.sample_contexts,
                        suggested_type = EXCLUDED.suggested_type,
                        confidence = EXCLUDED.confidence
                    """,
                    (
                        c.text,
                        c.document_ids,
                        c.document_count,
                        psycopg.types.json.Json(c.sample_contexts),
                        c.suggested_type,
                        c.confidence,
                    ),
                )
                stored += 1
            except Exception:
                pass
        conn.commit()

    return stored


def list_cross_doc_candidates(
    db_url: str,
    status: str = "pending",
    limit: int = 50,
) -> list[dict]:
    """List cross-document entity candidates.

    Returns list of dicts with candidate details.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        results = conn.execute(
            """
            SELECT id, text, document_count, sample_contexts,
                   suggested_type, confidence, status, created_at
            FROM cross_doc_entity_candidates
            WHERE status = %s
            ORDER BY document_count DESC, confidence DESC
            LIMIT %s
            """,
            (status, limit),
        ).fetchall()
        return [dict(r) for r in results]


def approve_cross_doc_candidate(db_url: str, candidate_id: str) -> str | None:
    """Approve a cross-doc candidate, creating it as a pending entity.

    Returns the pending entity ID, or None if failed.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        # Get candidate
        candidate = conn.execute(
            """
            SELECT text, document_ids, sample_contexts, suggested_type, confidence
            FROM cross_doc_entity_candidates
            WHERE id = %s AND status = 'pending'
            """,
            (candidate_id,),
        ).fetchone()

        if not candidate:
            return None

        # Get first document as source
        doc_ids = candidate["document_ids"]
        if not doc_ids:
            return None

        source_doc = conn.execute(
            "SELECT id FROM documents WHERE id = %s",
            (doc_ids[0],),
        ).fetchone()

        if not source_doc:
            return None

        # Create pending entity
        result = conn.execute(
            """
            INSERT INTO pending_entities
                (source_document_id, entity_name, entity_type, mentions, confidence, status)
            VALUES (%s, %s, %s, %s, %s, 'pending')
            RETURNING id
            """,
            (
                source_doc["id"],
                candidate["text"],
                candidate["suggested_type"] or "concept",
                psycopg.types.json.Json(candidate["sample_contexts"]),
                candidate["confidence"],
            ),
        ).fetchone()

        # Mark candidate as approved
        conn.execute(
            """
            UPDATE cross_doc_entity_candidates
            SET status = 'approved', reviewed_at = NOW()
            WHERE id = %s
            """,
            (candidate_id,),
        )

        conn.commit()
        return str(result["id"]) if result else None


def reject_cross_doc_candidate(db_url: str, candidate_id: str) -> bool:
    """Reject a cross-doc candidate.

    Returns True if successful.
    """
    with psycopg.connect(db_url) as conn:
        result = conn.execute(
            """
            UPDATE cross_doc_entity_candidates
            SET status = 'rejected', reviewed_at = NOW()
            WHERE id = %s AND status = 'pending'
            RETURNING id
            """,
            (candidate_id,),
        ).fetchone()
        conn.commit()
        return result is not None
