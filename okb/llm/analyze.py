"""Database/project-level analysis - aggregate entities and extract themes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row


@dataclass
class AnalysisResult:
    """Result from database/project analysis."""

    description: str
    topics: list[str]
    key_entities: list[dict]  # [{name, type, ref_count, doc_count}]
    stats: dict
    sample_count: int
    analyzed_at: datetime
    project: str | None = None


ANALYSIS_SYSTEM = """You are analyzing a personal knowledge base to understand its contents.
Based on the provided statistics, entity list, and document samples,
generate a concise description and topic keywords.

Focus on PURPOSE and THEMES, not statistics.
Good: "Technical notes on Python web development with Django and FastAPI"
Bad: "Contains 2500 code files and 63 markdown documents"

Return ONLY valid JSON with this structure:
{"description": "1-3 sentence description", "topics": ["topic1", "topic2", ...]}"""

ANALYSIS_USER = """## Content Statistics
{stats}

## Most Referenced Entities ({entity_count} shown)
{entity_list}

## Sample Document Titles and Excerpts ({sample_count} documents)
{document_samples}

---
Analyze this knowledge base and return JSON with "description" and "topics" fields.
The description should capture the overall purpose and main themes (1-3 sentences).
Topics should be 5-15 keywords that characterize the content."""


def get_entity_summary(
    db_url: str,
    project: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Get most-mentioned entities with reference counts.

    Returns list of dicts with: name, type, ref_count, doc_count
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        sql = """
            SELECT
                e.title as name,
                e.metadata->>'entity_type' as type,
                COUNT(DISTINCT r.id) as ref_count,
                COUNT(DISTINCT r.document_id) as doc_count
            FROM documents e
            JOIN entity_refs r ON r.entity_id = e.id
            WHERE e.source_type = 'entity'
        """
        params: list[Any] = []

        if project:
            sql += """
                AND r.document_id IN (
                    SELECT id FROM documents WHERE metadata->>'project' = %s
                )
            """
            params.append(project)

        sql += """
            GROUP BY e.id, e.title, e.metadata->>'entity_type'
            ORDER BY ref_count DESC
            LIMIT %s
        """
        params.append(limit)

        results = conn.execute(sql, params).fetchall()
        return [dict(r) for r in results]


def get_content_stats(db_url: str, project: str | None = None) -> dict:
    """Get content composition statistics.

    Returns dict with: source_types, projects, total_documents, total_tokens, date_range
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        # Base filter for project
        project_filter = ""
        params: list[Any] = []
        if project:
            project_filter = " WHERE metadata->>'project' = %s"
            params.append(project)

        # Source type breakdown
        sql = f"""
            SELECT
                source_type,
                COUNT(*) as doc_count,
                COALESCE(SUM(
                    (SELECT COALESCE(SUM(token_count), 0) FROM chunks WHERE document_id = d.id)
                ), 0) as token_count
            FROM documents d
            {project_filter}
            GROUP BY source_type
            ORDER BY doc_count DESC
        """
        source_types = conn.execute(sql, params).fetchall()

        # Total counts
        sql = f"""
            SELECT
                COUNT(*) as total_documents,
                (SELECT COUNT(*) FROM chunks c
                 JOIN documents d ON c.document_id = d.id
                 {project_filter.replace("WHERE", "WHERE" if project else "")}) as total_chunks
            FROM documents d
            {project_filter}
        """
        totals = conn.execute(sql, params + params if project else params).fetchone()

        # Project list (only if not filtering by project)
        projects = []
        if not project:
            sql = """
                SELECT DISTINCT metadata->>'project' as project
                FROM documents
                WHERE metadata->>'project' IS NOT NULL
                ORDER BY project
            """
            projects = [r["project"] for r in conn.execute(sql).fetchall()]

        # Date range
        sql = f"""
            SELECT
                MIN(COALESCE(
                    (metadata->>'document_date')::date,
                    (metadata->>'file_modified_at')::date,
                    created_at::date
                )) as earliest,
                MAX(COALESCE(
                    (metadata->>'document_date')::date,
                    (metadata->>'file_modified_at')::date,
                    created_at::date
                )) as latest
            FROM documents d
            {project_filter}
        """
        dates = conn.execute(sql, params).fetchone()

        total_tokens = sum(s["token_count"] for s in source_types)

        return {
            "source_types": {s["source_type"]: s["doc_count"] for s in source_types},
            "projects": projects,
            "total_documents": totals["total_documents"],
            "total_chunks": totals["total_chunks"],
            "total_tokens": total_tokens,
            "date_range": {
                "earliest": str(dates["earliest"]) if dates["earliest"] else None,
                "latest": str(dates["latest"]) if dates["latest"] else None,
            },
        }


def get_document_samples(
    db_url: str,
    project: str | None = None,
    sample_size: int = 15,
    strategy: str = "diverse",
) -> list[dict]:
    """Get representative document samples for topic extraction.

    Args:
        db_url: Database URL
        project: Filter by project
        sample_size: Number of documents to sample
        strategy: Sampling strategy - "diverse" uses embedding distance, "recent", or "random"

    Returns:
        List of dicts with: title, source_type, excerpt (first ~500 chars)
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        from pgvector.psycopg import register_vector

        register_vector(conn)

        project_filter = ""
        params: list[Any] = []
        if project:
            project_filter = " AND d.metadata->>'project' = %s"
            params.append(project)

        # Exclude derived documents
        base_filter = """
            d.source_path NOT LIKE '%%::todo/%%'
            AND d.source_path NOT LIKE 'okb://entity/%%'
            AND d.source_path NOT LIKE 'claude://%%'
        """

        if strategy == "diverse":
            # Use embedding-based diverse sampling:
            # Start with a random document, then iteratively pick documents
            # that are farthest from the already-selected set
            sql = f"""
                WITH first_doc AS (
                    SELECT d.id, d.title, d.source_type, d.content,
                           (SELECT embedding FROM chunks WHERE document_id = d.id LIMIT 1) as emb
                    FROM documents d
                    WHERE {base_filter} {project_filter}
                    ORDER BY RANDOM()
                    LIMIT 1
                )
                SELECT id, title, source_type,
                       LEFT(content, 500) as excerpt
                FROM first_doc
            """
            results = list(conn.execute(sql, params).fetchall())

            if results:
                # Get remaining documents with embeddings
                sql = f"""
                    SELECT d.id, d.title, d.source_type,
                           LEFT(d.content, 500) as excerpt,
                           (SELECT embedding FROM chunks WHERE document_id = d.id LIMIT 1) as emb
                    FROM documents d
                    WHERE {base_filter} {project_filter}
                    AND d.id != %s
                """
                remaining = list(conn.execute(sql, params + [results[0]["id"]]).fetchall())

                # Iteratively select documents farthest from selected set
                selected_embs = []
                if results[0].get("emb"):
                    # Get embedding for first doc
                    first_emb = conn.execute(
                        "SELECT embedding FROM chunks WHERE document_id = %s LIMIT 1",
                        (results[0]["id"],),
                    ).fetchone()
                    if first_emb:
                        selected_embs = [first_emb["embedding"]]

                while len(results) < sample_size and remaining:
                    if not selected_embs:
                        # No embeddings available, fall back to random
                        import random

                        next_doc = random.choice(remaining)
                        remaining.remove(next_doc)
                    else:
                        # Find doc with max min-distance from selected
                        best_doc = None
                        best_dist = -1
                        for doc in remaining:
                            if doc.get("emb") is None:
                                continue
                            # Min distance to any selected
                            min_dist = min(
                                1
                                - float(
                                    conn.execute(
                                        "SELECT %s::vector <=> %s::vector as dist",
                                        (doc["emb"], emb),
                                    ).fetchone()["dist"]
                                )
                                for emb in selected_embs
                            )
                            if min_dist > best_dist:
                                best_dist = min_dist
                                best_doc = doc

                        if best_doc is None:
                            break
                        next_doc = best_doc
                        remaining.remove(next_doc)
                        if next_doc.get("emb"):
                            selected_embs.append(next_doc["emb"])

                    results.append(
                        {
                            "id": next_doc["id"],
                            "title": next_doc["title"],
                            "source_type": next_doc["source_type"],
                            "excerpt": next_doc["excerpt"],
                        }
                    )

        elif strategy == "recent":
            sql = f"""
                SELECT d.title, d.source_type,
                       LEFT(d.content, 500) as excerpt
                FROM documents d
                WHERE {base_filter} {project_filter}
                ORDER BY d.updated_at DESC
                LIMIT %s
            """
            results = conn.execute(sql, params + [sample_size]).fetchall()

        else:  # random
            sql = f"""
                SELECT d.title, d.source_type,
                       LEFT(d.content, 500) as excerpt
                FROM documents d
                WHERE {base_filter} {project_filter}
                ORDER BY RANDOM()
                LIMIT %s
            """
            results = conn.execute(sql, params + [sample_size]).fetchall()

        return [
            {"title": r["title"], "source_type": r["source_type"], "excerpt": r["excerpt"]}
            for r in results
        ]


def analyze_database(
    db_url: str,
    project: str | None = None,
    sample_size: int = 15,
    auto_update: bool = True,
) -> AnalysisResult:
    """Run full database/project analysis.

    Args:
        db_url: Database URL
        project: Analyze specific project only
        sample_size: Number of documents to sample
        auto_update: Update database_metadata with results

    Returns:
        AnalysisResult with description, topics, entities, stats
    """
    from . import get_llm

    # Gather data
    stats = get_content_stats(db_url, project)
    entities = get_entity_summary(db_url, project, limit=20)
    samples = get_document_samples(db_url, project, sample_size, strategy="diverse")

    # Format stats for prompt
    stats_text = []
    stats_text.append(f"Total documents: {stats['total_documents']}")
    stats_text.append(f"Total tokens: ~{stats['total_tokens']:,}")
    if stats["source_types"]:
        types_str = ", ".join(
            f"{t}: {c}" for t, c in sorted(stats["source_types"].items(), key=lambda x: -x[1])
        )
        stats_text.append(f"Source types: {types_str}")
    if stats["projects"]:
        stats_text.append(f"Projects: {', '.join(stats['projects'])}")
    if stats["date_range"]["earliest"]:
        stats_text.append(
            f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}"
        )

    # Format entities for prompt
    entity_text = []
    for e in entities:
        entity_text.append(
            f"- {e['name']} ({e['type']}): {e['ref_count']} mentions in {e['doc_count']} docs"
        )
    if not entity_text:
        entity_text.append("(No entities extracted yet)")

    # Format samples for prompt
    sample_text = []
    for s in samples:
        excerpt = s["excerpt"].replace("\n", " ")[:200] if s["excerpt"] else ""
        sample_text.append(f"### {s['title']} ({s['source_type']})\n{excerpt}...")

    # Build prompt
    prompt = ANALYSIS_USER.format(
        stats="\n".join(stats_text),
        entity_count=len(entities),
        entity_list="\n".join(entity_text),
        sample_count=len(samples),
        document_samples="\n\n".join(sample_text),
    )

    # Call LLM
    llm = get_llm()
    if llm is None:
        raise RuntimeError(
            "No LLM provider configured. Analysis requires an LLM. "
            "Set ANTHROPIC_API_KEY or configure llm.provider in config."
        )

    response = llm.complete(prompt, system=ANALYSIS_SYSTEM, max_tokens=1024)

    # Parse response
    try:
        # Try to extract JSON from response
        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        data = json.loads(content)
        description = data.get("description", "")
        topics = data.get("topics", [])
    except (json.JSONDecodeError, KeyError):
        # Fallback: use raw response as description
        description = response.content.strip()[:500]
        topics = []

    result = AnalysisResult(
        description=description,
        topics=topics,
        key_entities=entities,
        stats=stats,
        sample_count=len(samples),
        analyzed_at=datetime.now(UTC),
        project=project,
    )

    # Update database metadata if requested
    if auto_update:
        _update_metadata(db_url, result)

    return result


def _update_metadata(db_url: str, result: AnalysisResult) -> None:
    """Update database_metadata with analysis results."""
    with psycopg.connect(db_url) as conn:
        # Use project-specific keys if analyzing a project
        suffix = f"_{result.project}" if result.project else ""

        conn.execute(
            """
            INSERT INTO database_metadata (key, value, source, updated_at)
            VALUES (%s, %s, 'llm', NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                source = 'llm',
                updated_at = NOW()
            """,
            (f"llm_description{suffix}", psycopg.types.json.Json(result.description)),
        )

        conn.execute(
            """
            INSERT INTO database_metadata (key, value, source, updated_at)
            VALUES (%s, %s, 'llm', NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                source = 'llm',
                updated_at = NOW()
            """,
            (f"llm_topics{suffix}", psycopg.types.json.Json(result.topics)),
        )

        # Also store analysis metadata
        conn.execute(
            """
            INSERT INTO database_metadata (key, value, source, updated_at)
            VALUES (%s, %s, 'llm', NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                source = 'llm',
                updated_at = NOW()
            """,
            (
                f"llm_analysis{suffix}",
                psycopg.types.json.Json(
                    {
                        "analyzed_at": result.analyzed_at.isoformat(),
                        "sample_count": result.sample_count,
                        "entity_count": len(result.key_entities),
                        "doc_count": result.stats["total_documents"],
                    }
                ),
            ),
        )

        conn.commit()


def format_analysis_result(result: AnalysisResult) -> str:
    """Format analysis result for display."""
    lines = ["## Knowledge Base Analysis\n"]

    if result.project:
        lines.append(f"**Project:** {result.project}\n")

    lines.append(f"**Description:** {result.description}\n")

    if result.topics:
        lines.append(f"**Topics:** {', '.join(result.topics)}\n")

    lines.append("\n### Content Statistics")
    lines.append(f"- Documents: {result.stats['total_documents']:,}")
    lines.append(f"- Tokens: ~{result.stats['total_tokens']:,}")
    if result.stats["source_types"]:
        sorted_types = sorted(result.stats["source_types"].items(), key=lambda x: -x[1])
        types_str = ", ".join(f"{t}: {c}" for t, c in sorted_types)
        lines.append(f"- Source types: {types_str}")
    if result.stats["projects"]:
        lines.append(f"- Projects: {', '.join(result.stats['projects'])}")

    if result.key_entities:
        lines.append("\n### Key Entities (by mentions)")
        for i, e in enumerate(result.key_entities[:10], 1):
            lines.append(
                f"{i}. {e['name']} ({e['type']}) - "
                f"{e['ref_count']} mentions in {e['doc_count']} docs"
            )

    timestamp = result.analyzed_at.strftime("%Y-%m-%d %H:%M")
    lines.append(f"\nAnalyzed {result.sample_count} document samples at {timestamp}")

    return "\n".join(lines)
