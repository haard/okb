"""Entity deduplication - find and merge duplicate entities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row


@dataclass
class EntityMergePair:
    """A pair of entities that may be duplicates."""

    canonical_id: str
    canonical_name: str
    canonical_type: str
    duplicate_id: str
    duplicate_name: str
    duplicate_type: str
    confidence: float
    reason: str  # "embedding_similarity", "alias_match", "llm"


DEDUP_SYSTEM_PROMPT = """\
You are an expert at identifying duplicate entities. Given a list of entity names and types,
identify groups that refer to the same real-world entity.

Consider:
- Abbreviations: "AWS" and "Amazon Web Services" are the same
- Spelling variations: "React.js", "ReactJS", "React" are the same
- Case differences: "Python" and "python" are the same
- With/without suffixes: "Google Inc" and "Google" are the same

Return ONLY valid JSON with this structure:
{
  "merge_groups": [
    {
      "canonical": "Full/preferred name",
      "duplicates": ["alias1", "alias2"],
      "confidence": 0.95,
      "reason": "Brief explanation"
    }
  ]
}

If no duplicates found, return: {"merge_groups": []}
"""

DEDUP_USER_PROMPT = """\
Analyze these entities for duplicates:

{entity_list}

Group any entities that refer to the same thing.
"""


def find_duplicate_entities(
    db_url: str,
    similarity_threshold: float = 0.85,
    use_llm: bool = True,
    entity_type: str | None = None,
    limit: int = 100,
) -> list[EntityMergePair]:
    """Find potential duplicate entities using embedding similarity and LLM.

    Args:
        db_url: Database URL
        similarity_threshold: Minimum cosine similarity to consider as duplicate
        use_llm: Whether to use LLM for batch deduplication
        entity_type: Filter to specific entity type
        limit: Maximum entities to analyze

    Returns:
        List of EntityMergePair objects representing potential duplicates
    """
    pairs: list[EntityMergePair] = []

    # Phase 1: Embedding similarity
    embedding_pairs = _find_by_embedding_similarity(
        db_url, similarity_threshold, entity_type, limit
    )
    pairs.extend(embedding_pairs)

    # Phase 2: Alias matching
    alias_pairs = _find_by_alias_match(db_url, entity_type)
    # Don't add if already found by embedding
    existing = {(p.canonical_id, p.duplicate_id) for p in pairs}
    existing.update({(p.duplicate_id, p.canonical_id) for p in pairs})
    for p in alias_pairs:
        if (p.canonical_id, p.duplicate_id) not in existing:
            pairs.append(p)
            existing.add((p.canonical_id, p.duplicate_id))
            existing.add((p.duplicate_id, p.canonical_id))

    # Phase 3: LLM batch identification
    if use_llm:
        llm_pairs = _find_by_llm(db_url, entity_type, limit)
        for p in llm_pairs:
            if (p.canonical_id, p.duplicate_id) not in existing:
                pairs.append(p)
                existing.add((p.canonical_id, p.duplicate_id))
                existing.add((p.duplicate_id, p.canonical_id))

    return pairs


def _find_by_embedding_similarity(
    db_url: str,
    threshold: float,
    entity_type: str | None,
    limit: int,
) -> list[EntityMergePair]:
    """Find duplicates by comparing entity embeddings."""
    pairs = []

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        from pgvector.psycopg import register_vector

        register_vector(conn)

        # Get entity documents with embeddings
        sql = """
            SELECT d.id, d.title, d.metadata->>'entity_type' as entity_type,
                   (SELECT embedding FROM chunks WHERE document_id = d.id LIMIT 1) as embedding
            FROM documents d
            WHERE d.source_type = 'entity'
        """
        params: list[Any] = []

        if entity_type:
            sql += " AND d.metadata->>'entity_type' = %s"
            params.append(entity_type)

        sql += " LIMIT %s"
        params.append(limit)

        entities = conn.execute(sql, params).fetchall()

        # Compare each pair
        for i, e1 in enumerate(entities):
            if e1["embedding"] is None:
                continue
            for e2 in entities[i + 1 :]:
                if e2["embedding"] is None:
                    continue

                # Calculate similarity
                result = conn.execute(
                    "SELECT 1 - (%s::vector <=> %s::vector) as similarity",
                    (e1["embedding"], e2["embedding"]),
                ).fetchone()
                similarity = result["similarity"]

                if similarity >= threshold:
                    # Prefer longer/more complete name as canonical
                    if len(e1["title"]) >= len(e2["title"]):
                        canonical, duplicate = e1, e2
                    else:
                        canonical, duplicate = e2, e1

                    pairs.append(
                        EntityMergePair(
                            canonical_id=str(canonical["id"]),
                            canonical_name=canonical["title"],
                            canonical_type=canonical["entity_type"] or "unknown",
                            duplicate_id=str(duplicate["id"]),
                            duplicate_name=duplicate["title"],
                            duplicate_type=duplicate["entity_type"] or "unknown",
                            confidence=similarity,
                            reason="embedding_similarity",
                        )
                    )

    return pairs


def _find_by_alias_match(
    db_url: str,
    entity_type: str | None,
) -> list[EntityMergePair]:
    """Find duplicates where one entity's name matches another's alias."""
    pairs = []

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        # Get entities with aliases
        sql = """
            SELECT d.id, d.title, d.metadata->>'entity_type' as entity_type,
                   d.metadata->'aliases' as aliases
            FROM documents d
            WHERE d.source_type = 'entity'
            AND d.metadata->'aliases' IS NOT NULL
        """
        params: list[Any] = []

        if entity_type:
            sql += " AND d.metadata->>'entity_type' = %s"
            params.append(entity_type)

        entities = conn.execute(sql, params).fetchall()

        # Build alias -> entity mapping
        alias_map: dict[str, dict] = {}
        for e in entities:
            aliases = e["aliases"] if isinstance(e["aliases"], list) else []
            for alias in aliases:
                if isinstance(alias, str):
                    normalized = alias.lower().strip()
                    alias_map[normalized] = e

        # Check if any entity name matches another's alias
        for e in entities:
            normalized_name = e["title"].lower().strip()
            if normalized_name in alias_map:
                other = alias_map[normalized_name]
                if other["id"] != e["id"]:
                    # Prefer the one with more aliases as canonical
                    e_aliases = e["aliases"] if isinstance(e["aliases"], list) else []
                    o_aliases = other["aliases"] if isinstance(other["aliases"], list) else []
                    if len(o_aliases) >= len(e_aliases):
                        canonical, duplicate = other, e
                    else:
                        canonical, duplicate = e, other

                    pairs.append(
                        EntityMergePair(
                            canonical_id=str(canonical["id"]),
                            canonical_name=canonical["title"],
                            canonical_type=canonical["entity_type"] or "unknown",
                            duplicate_id=str(duplicate["id"]),
                            duplicate_name=duplicate["title"],
                            duplicate_type=duplicate["entity_type"] or "unknown",
                            confidence=0.9,
                            reason="alias_match",
                        )
                    )

    return pairs


def _find_by_llm(
    db_url: str,
    entity_type: str | None,
    limit: int,
) -> list[EntityMergePair]:
    """Use LLM to identify duplicate entities in batch."""
    from .. import complete

    pairs = []

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        sql = """
            SELECT d.id, d.title, d.metadata->>'entity_type' as entity_type
            FROM documents d
            WHERE d.source_type = 'entity'
        """
        params: list[Any] = []

        if entity_type:
            sql += " AND d.metadata->>'entity_type' = %s"
            params.append(entity_type)

        sql += " ORDER BY d.title LIMIT %s"
        params.append(limit)

        entities = conn.execute(sql, params).fetchall()

        if len(entities) < 2:
            return []

        # Build entity list for prompt
        entity_lines = []
        entity_map = {}
        for e in entities:
            entity_lines.append(f"- {e['title']} ({e['entity_type']})")
            entity_map[e["title"].lower()] = e

        prompt = DEDUP_USER_PROMPT.format(entity_list="\n".join(entity_lines))

        response = complete(prompt, system=DEDUP_SYSTEM_PROMPT, max_tokens=2048, use_cache=True)

        if response is None:
            return []

        # Parse response
        try:
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            data = json.loads(content)
        except json.JSONDecodeError:
            return []

        merge_groups = data.get("merge_groups", [])
        for group in merge_groups:
            canonical_name = group.get("canonical", "")
            duplicates = group.get("duplicates", [])
            confidence = group.get("confidence", 0.8)

            canonical = entity_map.get(canonical_name.lower())
            if not canonical:
                continue

            for dup_name in duplicates:
                dup = entity_map.get(dup_name.lower())
                if dup and dup["id"] != canonical["id"]:
                    pairs.append(
                        EntityMergePair(
                            canonical_id=str(canonical["id"]),
                            canonical_name=canonical["title"],
                            canonical_type=canonical["entity_type"] or "unknown",
                            duplicate_id=str(dup["id"]),
                            duplicate_name=dup["title"],
                            duplicate_type=dup["entity_type"] or "unknown",
                            confidence=confidence,
                            reason="llm",
                        )
                    )

    return pairs


def create_pending_merge(db_url: str, pair: EntityMergePair) -> str | None:
    """Create a pending merge proposal.

    Returns the merge ID, or None if already exists.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        try:
            result = conn.execute(
                """
                INSERT INTO pending_entity_merges
                    (canonical_id, duplicate_id, confidence, reason, status)
                VALUES (%s, %s, %s, %s, 'pending')
                ON CONFLICT (canonical_id, duplicate_id) DO NOTHING
                RETURNING id
                """,
                (pair.canonical_id, pair.duplicate_id, pair.confidence, pair.reason),
            ).fetchone()
            conn.commit()
            return str(result["id"]) if result else None
        except Exception:
            return None


def execute_merge(db_url: str, canonical_id: str, duplicate_id: str) -> bool:
    """Execute a merge: redirect refs from duplicate to canonical, add alias, delete duplicate.

    Args:
        db_url: Database URL
        canonical_id: ID of the entity to keep
        duplicate_id: ID of the entity to merge into canonical

    Returns:
        True if merge succeeded, False otherwise
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        try:
            # Get duplicate info for alias
            duplicate = conn.execute(
                "SELECT title FROM documents WHERE id = %s AND source_type = 'entity'",
                (duplicate_id,),
            ).fetchone()

            if not duplicate:
                return False

            # 1. Redirect all entity_refs from duplicate to canonical
            conn.execute(
                """
                UPDATE entity_refs
                SET entity_id = %s
                WHERE entity_id = %s
                """,
                (canonical_id, duplicate_id),
            )

            # 2. Add duplicate's name as alias of canonical
            conn.execute(
                """
                INSERT INTO entity_aliases (alias_text, entity_id, confidence, source)
                VALUES (%s, %s, 1.0, 'merge')
                ON CONFLICT (alias_text, entity_id) DO NOTHING
                """,
                (duplicate["title"], canonical_id),
            )

            # 3. Also copy any existing aliases from duplicate to canonical
            conn.execute(
                """
                INSERT INTO entity_aliases (alias_text, entity_id, confidence, source)
                SELECT alias_text, %s, confidence, 'merge'
                FROM entity_aliases WHERE entity_id = %s
                ON CONFLICT (alias_text, entity_id) DO NOTHING
                """,
                (canonical_id, duplicate_id),
            )

            # 4. Delete duplicate entity document (cascades to chunks)
            conn.execute(
                "DELETE FROM documents WHERE id = %s",
                (duplicate_id,),
            )

            conn.commit()
            return True

        except Exception:
            return False


def approve_merge(db_url: str, merge_id: str) -> bool:
    """Approve and execute a pending merge.

    Returns True if successful.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        # Get merge details
        merge = conn.execute(
            """
            SELECT canonical_id, duplicate_id
            FROM pending_entity_merges
            WHERE id = %s AND status = 'pending'
            """,
            (merge_id,),
        ).fetchone()

        if not merge:
            return False

        # Execute the merge
        if execute_merge(db_url, str(merge["canonical_id"]), str(merge["duplicate_id"])):
            # Mark as approved
            conn.execute(
                """
                UPDATE pending_entity_merges
                SET status = 'approved', reviewed_at = NOW()
                WHERE id = %s
                """,
                (merge_id,),
            )
            conn.commit()
            return True

        return False


def reject_merge(db_url: str, merge_id: str) -> bool:
    """Reject a pending merge.

    Returns True if successful.
    """
    with psycopg.connect(db_url) as conn:
        result = conn.execute(
            """
            UPDATE pending_entity_merges
            SET status = 'rejected', reviewed_at = NOW()
            WHERE id = %s AND status = 'pending'
            RETURNING id
            """,
            (merge_id,),
        ).fetchone()
        conn.commit()
        return result is not None


def list_pending_merges(
    db_url: str,
    limit: int = 50,
) -> list[dict]:
    """List pending entity merge proposals.

    Returns list of dicts with merge details.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        results = conn.execute(
            """
            SELECT
                m.id, m.confidence, m.reason, m.detected_at,
                c.id as canonical_id, c.title as canonical_name,
                c.metadata->>'entity_type' as canonical_type,
                d.id as duplicate_id, d.title as duplicate_name,
                d.metadata->>'entity_type' as duplicate_type
            FROM pending_entity_merges m
            JOIN documents c ON c.id = m.canonical_id
            JOIN documents d ON d.id = m.duplicate_id
            WHERE m.status = 'pending'
            ORDER BY m.confidence DESC, m.detected_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in results]
