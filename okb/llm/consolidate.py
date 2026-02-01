"""Entity consolidation orchestration - clustering, relationships, and full pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row


@dataclass
class TopicCluster:
    """A cluster of related entities and documents."""

    id: str
    name: str
    description: str | None
    member_count: int
    entities: list[dict]  # [{id, name, type, distance}]
    documents: list[dict]  # [{id, title, distance}]


@dataclass
class EntityRelationship:
    """A relationship between two entities."""

    id: str
    source_entity: dict  # {id, name, type}
    target_entity: dict  # {id, name, type}
    relationship_type: str  # works_for, uses, belongs_to, related_to
    confidence: float
    context: str | None


@dataclass
class ConsolidationResult:
    """Result from running consolidation pipeline."""

    duplicates_found: int = 0
    merges_pending: int = 0
    merges_auto_approved: int = 0
    cross_doc_candidates: int = 0
    clusters_created: int = 0
    relationships_found: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    errors: list[str] = field(default_factory=list)


RELATIONSHIP_SYSTEM = """\
You are an expert at identifying relationships between entities.
Given pairs of entities that appear in the same documents, identify any relationships.

Relationship types:
- works_for: Person works for organization
- uses: Project/person uses technology
- belongs_to: Entity belongs to/is part of another
- related_to: General association (only if no specific type applies)

Return ONLY valid JSON:
{
  "relationships": [
    {
      "source": "Entity 1",
      "target": "Entity 2",
      "type": "uses",
      "confidence": 0.9,
      "reason": "Brief explanation"
    }
  ]
}

If no relationships found, return: {"relationships": []}
"""

RELATIONSHIP_USER = """\
Identify relationships between these entity pairs that co-occur in documents:

{entity_pairs}

Only include high-confidence relationships.
"""

CLUSTER_NAMING_SYSTEM = """\
You are naming a topic cluster based on its member entities.
Create a short, descriptive name (2-5 words) and brief description.

Return ONLY valid JSON:
{"name": "Cluster Name", "description": "One sentence description"}
"""

CLUSTER_NAMING_USER = """\
Name this cluster containing these entities:

{entity_list}

The cluster should have a name that captures the common theme.
"""


def run_consolidation(
    db_url: str,
    detect_duplicates: bool = True,
    detect_cross_doc: bool = True,
    build_clusters: bool = True,
    extract_relationships: bool = True,
    auto_merge_threshold: float = 0.95,
    dry_run: bool = False,
) -> ConsolidationResult:
    """Run the full entity consolidation pipeline.

    Args:
        db_url: Database URL
        detect_duplicates: Run duplicate detection
        detect_cross_doc: Run cross-document entity detection
        build_clusters: Build topic clusters
        extract_relationships: Extract entity relationships
        auto_merge_threshold: Auto-approve merges above this confidence
        dry_run: Don't make changes, just report what would happen

    Returns:
        ConsolidationResult with counts and status
    """
    result = ConsolidationResult()

    # Log the run
    run_id = None
    if not dry_run:
        with psycopg.connect(db_url) as conn:
            r = conn.execute(
                "INSERT INTO consolidation_runs (run_type) VALUES ('full') RETURNING id"
            ).fetchone()
            run_id = r[0] if r else None
            conn.commit()

    try:
        # Phase 1: Duplicate detection
        if detect_duplicates:
            from .extractors.dedup import (
                approve_merge,
                create_pending_merge,
                find_duplicate_entities,
            )

            pairs = find_duplicate_entities(db_url)
            result.duplicates_found = len(pairs)

            if not dry_run:
                for pair in pairs:
                    if pair.confidence >= auto_merge_threshold:
                        # Auto-approve high-confidence merges
                        merge_id = create_pending_merge(db_url, pair)
                        if merge_id and approve_merge(db_url, merge_id):
                            result.merges_auto_approved += 1
                    else:
                        # Create pending for review
                        if create_pending_merge(db_url, pair):
                            result.merges_pending += 1

        # Phase 2: Cross-document detection
        if detect_cross_doc:
            from .extractors.cross_doc import (
                classify_candidates,
                find_cross_document_entities,
                store_candidates,
            )

            candidates = find_cross_document_entities(db_url)
            if candidates:
                # Classify with LLM
                classify_candidates(candidates, db_url)
                result.cross_doc_candidates = len(candidates)

                if not dry_run:
                    store_candidates(db_url, candidates)

        # Phase 3: Topic clustering
        if build_clusters:
            clusters = build_topic_clusters(db_url, dry_run=dry_run)
            result.clusters_created = len(clusters)

        # Phase 4: Entity relationships
        if extract_relationships:
            relationships = extract_entity_relationships(db_url, dry_run=dry_run)
            result.relationships_found = len(relationships)

        result.completed_at = datetime.now(UTC)

        # Update run record
        if run_id and not dry_run:
            with psycopg.connect(db_url) as conn:
                conn.execute(
                    """
                    UPDATE consolidation_runs
                    SET completed_at = NOW(),
                        stats = %s
                    WHERE id = %s
                    """,
                    (
                        psycopg.types.json.Json(
                            {
                                "duplicates_found": result.duplicates_found,
                                "merges_pending": result.merges_pending,
                                "merges_auto_approved": result.merges_auto_approved,
                                "cross_doc_candidates": result.cross_doc_candidates,
                                "clusters_created": result.clusters_created,
                                "relationships_found": result.relationships_found,
                            }
                        ),
                        run_id,
                    ),
                )
                conn.commit()

    except Exception as e:
        result.errors.append(str(e))
        if run_id and not dry_run:
            with psycopg.connect(db_url) as conn:
                conn.execute(
                    "UPDATE consolidation_runs SET error = %s WHERE id = %s",
                    (str(e), run_id),
                )
                conn.commit()

    return result


def build_topic_clusters(
    db_url: str,
    n_clusters: int | None = None,
    min_cluster_size: int = 3,
    dry_run: bool = False,
) -> list[TopicCluster]:
    """Build topic clusters from entity embeddings using k-means.

    Args:
        db_url: Database URL
        n_clusters: Number of clusters (auto-determined if None)
        min_cluster_size: Minimum entities per cluster
        dry_run: Don't save to database

    Returns:
        List of TopicCluster objects
    """
    from . import complete

    clusters: list[TopicCluster] = []

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        from pgvector.psycopg import register_vector

        register_vector(conn)

        # Get entity embeddings
        entities = conn.execute(
            """
            SELECT d.id, d.title, d.metadata->>'entity_type' as entity_type,
                   (SELECT embedding FROM chunks WHERE document_id = d.id LIMIT 1) as embedding
            FROM documents d
            WHERE d.source_type = 'entity'
            AND EXISTS (SELECT 1 FROM chunks WHERE document_id = d.id)
            """
        ).fetchall()

        if len(entities) < min_cluster_size:
            return []

        # Auto-determine cluster count: sqrt(n/2), min 2, max 20
        if n_clusters is None:
            n_clusters = max(2, min(20, int((len(entities) / 2) ** 0.5)))

        # Simple k-means using PostgreSQL
        # Initialize centroids with random entities
        import random

        centroid_entities = random.sample(list(entities), min(n_clusters, len(entities)))
        centroids = [e["embedding"] for e in centroid_entities if e["embedding"]]

        if len(centroids) < 2:
            return []

        # Assign entities to nearest centroid
        entity_clusters: dict[int, list[dict]] = {i: [] for i in range(len(centroids))}

        for entity in entities:
            if entity["embedding"] is None:
                continue

            # Find nearest centroid
            best_cluster = 0
            best_distance = float("inf")
            for i, centroid in enumerate(centroids):
                result = conn.execute(
                    "SELECT %s::vector <=> %s::vector as dist",
                    (entity["embedding"], centroid),
                ).fetchone()
                dist = result["dist"]
                if dist < best_distance:
                    best_distance = dist
                    best_cluster = i

            entity_clusters[best_cluster].append(
                {
                    "id": str(entity["id"]),
                    "name": entity["title"],
                    "type": entity["entity_type"],
                    "distance": best_distance,
                    "embedding": entity["embedding"],
                }
            )

        # Filter clusters by size and create
        for cluster_idx, members in entity_clusters.items():
            if len(members) < min_cluster_size:
                continue

            # Calculate centroid as average of member embeddings
            if not members:
                continue

            # Get cluster name from LLM
            entity_list = "\n".join(
                f"- {m['name']} ({m['type']})" for m in sorted(members, key=lambda x: x["distance"])
            )
            prompt = CLUSTER_NAMING_USER.format(entity_list=entity_list)
            response = complete(prompt, system=CLUSTER_NAMING_SYSTEM, max_tokens=256, use_cache=True)

            cluster_name = f"Cluster {cluster_idx + 1}"
            cluster_desc = None
            if response:
                try:
                    content = response.content.strip()
                    if content.startswith("```"):
                        lines = content.split("\n")
                        content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                    data = json.loads(content)
                    cluster_name = data.get("name", cluster_name)
                    cluster_desc = data.get("description")
                except json.JSONDecodeError:
                    pass

            # Calculate centroid (average embedding)
            # For simplicity, use first member's embedding as proxy
            centroid = members[0]["embedding"]

            cluster = TopicCluster(
                id="",
                name=cluster_name,
                description=cluster_desc,
                member_count=len(members),
                entities=[
                    {"id": m["id"], "name": m["name"], "type": m["type"], "distance": m["distance"]}
                    for m in members
                ],
                documents=[],
            )

            if not dry_run:
                # Save cluster
                result = conn.execute(
                    """
                    INSERT INTO topic_clusters (name, description, centroid, member_count)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (cluster_name, cluster_desc, centroid, len(members)),
                ).fetchone()

                if result:
                    cluster.id = str(result["id"])

                    # Add members
                    for m in members:
                        conn.execute(
                            """
                            INSERT INTO topic_cluster_members
                                (cluster_id, document_id, distance, is_entity)
                            VALUES (%s, %s, %s, TRUE)
                            ON CONFLICT DO NOTHING
                            """,
                            (result["id"], m["id"], m["distance"]),
                        )

                conn.commit()

            clusters.append(cluster)

    return clusters


def extract_entity_relationships(
    db_url: str,
    entity_ids: list[str] | None = None,
    dry_run: bool = False,
) -> list[EntityRelationship]:
    """Extract relationships between entities that co-occur in documents.

    Args:
        db_url: Database URL
        entity_ids: Filter to specific entities (None = all)
        dry_run: Don't save to database

    Returns:
        List of EntityRelationship objects
    """
    from . import complete

    relationships: list[EntityRelationship] = []

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        # Find entity pairs that co-occur in documents
        sql = """
            SELECT
                e1.id as e1_id, e1.title as e1_name,
                e1.metadata->>'entity_type' as e1_type,
                e2.id as e2_id, e2.title as e2_name,
                e2.metadata->>'entity_type' as e2_type,
                COUNT(DISTINCT r1.document_id) as shared_docs
            FROM entity_refs r1
            JOIN entity_refs r2 ON r1.document_id = r2.document_id
            JOIN documents e1 ON e1.id = r1.entity_id
            JOIN documents e2 ON e2.id = r2.entity_id
            WHERE r1.entity_id < r2.entity_id  -- Avoid duplicates
            AND e1.source_type = 'entity'
            AND e2.source_type = 'entity'
        """
        params: list[Any] = []

        if entity_ids:
            sql += " AND (r1.entity_id = ANY(%s) OR r2.entity_id = ANY(%s))"
            params.extend([entity_ids, entity_ids])

        sql += """
            GROUP BY e1.id, e1.title, e1.metadata->>'entity_type',
                     e2.id, e2.title, e2.metadata->>'entity_type'
            HAVING COUNT(DISTINCT r1.document_id) >= 2
            ORDER BY shared_docs DESC
            LIMIT 50
        """

        pairs = conn.execute(sql, params).fetchall()

        if not pairs:
            return []

        # Format for LLM
        pair_lines = []
        pair_map = {}
        for p in pairs:
            key = f"{p['e1_name']}|{p['e2_name']}"
            pair_lines.append(
                f"- {p['e1_name']} ({p['e1_type']}) <-> "
                f"{p['e2_name']} ({p['e2_type']}) "
                f"[{p['shared_docs']} shared docs]"
            )
            pair_map[key] = p

        prompt = RELATIONSHIP_USER.format(entity_pairs="\n".join(pair_lines))
        response = complete(prompt, system=RELATIONSHIP_SYSTEM, max_tokens=2048, use_cache=True)

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

        for rel in data.get("relationships", []):
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            rel_type = rel.get("type", "related_to")
            confidence = rel.get("confidence", 0.5)
            reason = rel.get("reason")

            # Find matching pair
            key = f"{source_name}|{target_name}"
            key_rev = f"{target_name}|{source_name}"
            pair = pair_map.get(key) or pair_map.get(key_rev)

            if pair:
                # Ensure source/target order matches pair
                if key_rev in pair_map:
                    source_name, target_name = target_name, source_name

                relationship = EntityRelationship(
                    id="",
                    source_entity={
                        "id": str(pair["e1_id"]),
                        "name": pair["e1_name"],
                        "type": pair["e1_type"],
                    },
                    target_entity={
                        "id": str(pair["e2_id"]),
                        "name": pair["e2_name"],
                        "type": pair["e2_type"],
                    },
                    relationship_type=rel_type,
                    confidence=confidence,
                    context=reason,
                )

                if not dry_run:
                    result = conn.execute(
                        """
                        INSERT INTO entity_relationships
                            (source_entity_id, target_entity_id, relationship_type,
                             confidence, context)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (source_entity_id, target_entity_id, relationship_type)
                        DO UPDATE SET confidence = EXCLUDED.confidence
                        RETURNING id
                        """,
                        (pair["e1_id"], pair["e2_id"], rel_type, confidence, reason),
                    ).fetchone()
                    if result:
                        relationship.id = str(result["id"])

                relationships.append(relationship)

        if not dry_run:
            conn.commit()

    return relationships


def get_topic_clusters(db_url: str, limit: int = 20) -> list[dict]:
    """Get topic clusters with their members.

    Returns list of cluster dicts.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        clusters = conn.execute(
            """
            SELECT id, name, description, member_count, created_at
            FROM topic_clusters
            ORDER BY member_count DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()

        result = []
        for c in clusters:
            # Get members
            members = conn.execute(
                """
                SELECT d.id, d.title, d.source_type, m.distance, m.is_entity
                FROM topic_cluster_members m
                JOIN documents d ON d.id = m.document_id
                WHERE m.cluster_id = %s
                ORDER BY m.distance
                LIMIT 20
                """,
                (c["id"],),
            ).fetchall()

            result.append(
                {
                    "id": str(c["id"]),
                    "name": c["name"],
                    "description": c["description"],
                    "member_count": c["member_count"],
                    "members": [
                        {
                            "id": str(m["id"]),
                            "title": m["title"],
                            "type": m["source_type"],
                            "is_entity": m["is_entity"],
                            "distance": m["distance"],
                        }
                        for m in members
                    ],
                }
            )

        return result


def get_entity_relationships(
    db_url: str,
    entity_name: str | None = None,
    relationship_type: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Get entity relationships.

    Args:
        db_url: Database URL
        entity_name: Filter to relationships involving this entity
        relationship_type: Filter by relationship type (works_for, uses, belongs_to, related_to)
        limit: Maximum results

    Returns:
        List of relationship dicts.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        sql = """
            SELECT
                r.id, r.relationship_type, r.confidence, r.context,
                s.id as source_id, s.title as source_name,
                s.metadata->>'entity_type' as source_type,
                t.id as target_id, t.title as target_name,
                t.metadata->>'entity_type' as target_type
            FROM entity_relationships r
            JOIN documents s ON s.id = r.source_entity_id
            JOIN documents t ON t.id = r.target_entity_id
            WHERE 1=1
        """
        params: list[Any] = []

        if entity_name:
            sql += " AND (LOWER(s.title) = LOWER(%s) OR LOWER(t.title) = LOWER(%s))"
            params.extend([entity_name, entity_name])

        if relationship_type:
            sql += " AND r.relationship_type = %s"
            params.append(relationship_type)

        sql += " ORDER BY r.confidence DESC LIMIT %s"
        params.append(limit)

        results = conn.execute(sql, params).fetchall()

        return [
            {
                "id": str(r["id"]),
                "source": {
                    "id": str(r["source_id"]),
                    "name": r["source_name"],
                    "type": r["source_type"],
                },
                "target": {
                    "id": str(r["target_id"]),
                    "name": r["target_name"],
                    "type": r["target_type"],
                },
                "type": r["relationship_type"],
                "confidence": r["confidence"],
                "context": r["context"],
            }
            for r in results
        ]


def format_consolidation_result(result: ConsolidationResult) -> str:
    """Format consolidation result for display."""
    lines = ["## Consolidation Results\n"]

    if result.duplicates_found:
        lines.append(f"**Duplicate Detection:** {result.duplicates_found} potential duplicates found")
        if result.merges_auto_approved:
            lines.append(f"  - {result.merges_auto_approved} auto-approved (high confidence)")
        if result.merges_pending:
            lines.append(f"  - {result.merges_pending} pending review")

    if result.cross_doc_candidates:
        lines.append(
            f"**Cross-Document Entities:** {result.cross_doc_candidates} candidates detected"
        )

    if result.clusters_created:
        lines.append(f"**Topic Clusters:** {result.clusters_created} clusters created")

    if result.relationships_found:
        lines.append(f"**Entity Relationships:** {result.relationships_found} relationships found")

    if result.errors:
        lines.append(f"\n**Errors:** {len(result.errors)}")
        for err in result.errors[:5]:
            lines.append(f"  - {err}")

    if result.completed_at:
        duration = (result.completed_at - result.started_at).total_seconds()
        lines.append(f"\nCompleted in {duration:.1f}s")

    return "\n".join(lines)
