"""Document enrichment orchestration - extract TODOs and entities from documents."""

from __future__ import annotations

import hashlib
import sys
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .extractors import (
    EnrichmentResult,
    ExtractedEntity,
    ExtractedTodo,
    extract_entities,
    extract_todos,
)
from .extractors.entity import entity_source_path

# Global thread pool for async embedding operations
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the global thread pool for async operations."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="enrich-embed")
    return _executor


def shutdown_executor(wait: bool = True) -> None:
    """Shutdown the global executor. Call at end of session."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None


def _get_embedder(use_modal: bool = False):
    """Get embedder based on configuration.

    Args:
        use_modal: If True, try to use Modal GPU embedder; fall back to local on failure

    Returns:
        Embedder object with embed_document(text) method
    """
    if use_modal:
        try:
            import modal

            modal_embedder = modal.Cls.from_name("knowledge-embedder", "Embedder")()

            class ModalEmbedderWrapper:
                """Wrapper to provide consistent interface."""

                def __init__(self, embedder):
                    self._embedder = embedder

                def embed_document(self, text: str) -> list[float]:
                    return self._embedder.embed_single.remote(text, is_query=False)

            return ModalEmbedderWrapper(modal_embedder)
        except Exception as e:
            print(f"Modal unavailable ({e}), using local CPU embedding", file=sys.stderr)

    # Fall back to local embedder
    from ..local_embedder import embed_document

    class LocalEmbedderWrapper:
        """Wrapper to provide consistent interface."""

        def embed_document(self, text: str) -> list[float]:
            return embed_document(text)

    return LocalEmbedderWrapper()


@dataclass
class EnrichmentConfig:
    """Configuration for document enrichment."""

    enabled: bool = True
    version: int = 1

    # What to extract
    extract_todos: bool = True
    extract_entities: bool = True

    # Auto-create behavior
    auto_create_todos: bool = True
    auto_create_entities: bool = False  # Entities go to pending by default

    # Confidence thresholds
    min_confidence_todo: float = 0.7
    min_confidence_entity: float = 0.8

    # Source types to auto-enrich during ingest
    auto_enrich_source_types: set[str] = field(
        default_factory=lambda: {"markdown", "org", "text"}
    )

    @classmethod
    def from_config(cls, cfg: dict) -> EnrichmentConfig:
        """Create from config dict."""
        auto_enrich = cfg.get("auto_enrich", {})
        auto_enrich_types = {k for k, v in auto_enrich.items() if v}

        default_types = {"markdown", "org", "text"}
        return cls(
            enabled=cfg.get("enabled", True),
            version=cfg.get("version", 1),
            extract_todos=cfg.get("extract_todos", True),
            extract_entities=cfg.get("extract_entities", True),
            auto_create_todos=cfg.get("auto_create_todos", True),
            auto_create_entities=cfg.get("auto_create_entities", False),
            min_confidence_todo=cfg.get("min_confidence_todo", 0.7),
            min_confidence_entity=cfg.get("min_confidence_entity", 0.8),
            auto_enrich_source_types=auto_enrich_types if auto_enrich_types else default_types,
        )


def enrich_document(
    content: str,
    title: str,
    source_type: str,
    config: EnrichmentConfig | None = None,
) -> EnrichmentResult:
    """Enrich a single document with extracted TODOs and entities.

    Args:
        content: Document content
        title: Document title
        source_type: Source type of the document
        config: Enrichment configuration

    Returns:
        EnrichmentResult with extracted TODOs and entities
    """
    if config is None:
        config = EnrichmentConfig()

    result = EnrichmentResult()

    if config.extract_todos:
        result.todos = extract_todos(
            content=content,
            title=title,
            source_type=source_type,
            min_confidence=config.min_confidence_todo,
        )

    if config.extract_entities:
        result.entities = extract_entities(
            content=content,
            title=title,
            source_type=source_type,
            min_confidence=config.min_confidence_entity,
        )

    return result


def _create_todo_document(
    todo: ExtractedTodo,
    parent_source_path: str,
    parent_title: str,
    db_url: str,
    project: str | None = None,
    use_modal: bool = False,
) -> str | None:
    """Create a TODO document from an extracted TODO.

    Returns the source_path of the created document, or None if creation failed.
    """
    embedder = _get_embedder(use_modal)

    # Generate unique source path
    todo_id = str(uuid.uuid4())[:8]
    source_path = f"{parent_source_path}::todo/{todo_id}"

    # Build content
    content = todo.title
    if todo.content:
        content += f"\n\n{todo.content}"
    if todo.source_context:
        content += f"\n\n---\nExtracted from: {todo.source_context}"

    # Build metadata
    metadata: dict[str, Any] = {
        "source": "enrichment",
        "parent_document": parent_source_path,
        "parent_title": parent_title,
    }
    if project:
        metadata["project"] = project
    if todo.assignee:
        metadata["assignee"] = todo.assignee

    # Content hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Build embedding text
    embedding_parts = [f"TODO: {todo.title}"]
    if project:
        embedding_parts.append(f"Project: {project}")
    if todo.content:
        embedding_parts.append(f"Details: {todo.content}")
    embedding_text = "\n".join(embedding_parts)

    embedding = embedder.embed_document(embedding_text)

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        try:
            doc_id = conn.execute(
                """
                INSERT INTO documents (
                    source_path, source_type, title, content, metadata, content_hash,
                    status, priority, due_date
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                (
                    source_path,
                    "enriched-todo",
                    todo.title,
                    content,
                    psycopg.types.json.Json(metadata),
                    content_hash,
                    "pending",
                    todo.priority,
                    todo.due_date,
                ),
            ).fetchone()

            if doc_id is None:
                return None

            # Insert chunk
            token_count = len(content) // 4
            conn.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, content, embedding_text, embedding, token_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    doc_id["id"],
                    0,
                    content,
                    embedding_text,
                    embedding,
                    token_count,
                    psycopg.types.json.Json({}),
                ),
            )
            conn.commit()
            return source_path
        except Exception as e:
            print(f"Error creating TODO document: {e}", file=sys.stderr)
            return None


def _create_pending_entity(
    entity: ExtractedEntity,
    source_document_id: str,
    db_url: str,
) -> str | None:
    """Create a pending entity suggestion.

    Returns the pending entity ID, or None if creation failed.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        try:
            result = conn.execute(
                """
                INSERT INTO pending_entities (
                    source_document_id, entity_name, entity_type, aliases,
                    description, mentions, confidence, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
                RETURNING id
                """,
                (
                    source_document_id,
                    entity.name,
                    entity.entity_type,
                    psycopg.types.json.Json(entity.aliases),
                    entity.description,
                    psycopg.types.json.Json(entity.mentions),
                    entity.confidence,
                ),
            ).fetchone()
            conn.commit()
            return str(result["id"]) if result else None
        except Exception as e:
            print(f"Error creating pending entity: {e}", file=sys.stderr)
            return None


def _create_entity_document(
    entity: ExtractedEntity,
    source_document_id: str,
    db_url: str,
    use_modal: bool = False,
) -> str | None:
    """Create an entity document and add entity_refs.

    Returns the source_path of the created/existing entity document.
    """
    embedder = _get_embedder(use_modal)

    source_path = entity_source_path(entity.entity_type, entity.name)

    # Build content
    content_parts = [f"# {entity.name}"]
    content_parts.append(f"Type: {entity.entity_type}")
    if entity.aliases:
        content_parts.append(f"Also known as: {', '.join(entity.aliases)}")
    if entity.description:
        content_parts.append(f"\n{entity.description}")
    content = "\n".join(content_parts)

    # Build metadata
    metadata = {
        "source": "enrichment",
        "entity_type": entity.entity_type,
        "aliases": entity.aliases,
    }

    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Build embedding text
    embedding_parts = [f"Entity: {entity.name}", f"Type: {entity.entity_type}"]
    if entity.aliases:
        embedding_parts.append(f"Aliases: {', '.join(entity.aliases)}")
    if entity.description:
        embedding_parts.append(entity.description)
    embedding_text = "\n".join(embedding_parts)

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        try:
            # Check if entity document already exists
            existing = conn.execute(
                "SELECT id FROM documents WHERE source_path = %s",
                (source_path,),
            ).fetchone()

            if existing:
                entity_doc_id = existing["id"]
                # Update existing document
                conn.execute(
                    """
                    UPDATE documents SET
                        content = %s,
                        metadata = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (content, psycopg.types.json.Json(metadata), entity_doc_id),
                )
            else:
                # Create new entity document
                embedding = embedder.embed_document(embedding_text)

                result = conn.execute(
                    """
                    INSERT INTO documents (
                        source_path, source_type, title, content, metadata, content_hash
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        source_path,
                        "entity",
                        entity.name,
                        content,
                        psycopg.types.json.Json(metadata),
                        content_hash,
                    ),
                ).fetchone()

                entity_doc_id = result["id"]

                # Insert chunk
                token_count = len(content) // 4
                conn.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding_text, embedding, token_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entity_doc_id,
                        0,
                        content,
                        embedding_text,
                        embedding,
                        token_count,
                        psycopg.types.json.Json({}),
                    ),
                )

            # Add entity reference linking entity to source document
            for mention in entity.mentions or [entity.name]:
                conn.execute(
                    """
                    INSERT INTO entity_refs (entity_id, document_id, mention_text, confidence)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (entity_id, document_id, mention_text) DO NOTHING
                    """,
                    (entity_doc_id, source_document_id, mention[:500], entity.confidence),
                )

            conn.commit()
            return source_path
        except Exception as e:
            print(f"Error creating entity document: {e}", file=sys.stderr)
            return None


def process_enrichment(
    document_id: str,
    source_path: str,
    title: str,
    content: str,
    source_type: str,
    db_url: str,
    config: EnrichmentConfig | None = None,
    project: str | None = None,
    use_modal: bool = False,
) -> dict:
    """Run enrichment on a document and store results.

    Args:
        document_id: UUID of the document
        source_path: Source path of the document
        title: Document title
        content: Document content
        source_type: Type of document
        db_url: Database URL
        config: Enrichment configuration
        project: Project name for TODOs
        use_modal: If True, use Modal GPU for embedding; else local CPU

    Returns:
        Dict with counts: {todos_created, entities_pending, entities_created}
    """
    if config is None:
        config = EnrichmentConfig()

    result = enrich_document(content, title, source_type, config)

    stats = {
        "todos_created": 0,
        "entities_pending": 0,
        "entities_created": 0,
    }

    # Process TODOs
    if result.todos and config.auto_create_todos:
        for todo in result.todos:
            if _create_todo_document(todo, source_path, title, db_url, project, use_modal):
                stats["todos_created"] += 1

    # Process entities
    for entity in result.entities:
        if config.auto_create_entities:
            if _create_entity_document(entity, document_id, db_url, use_modal):
                stats["entities_created"] += 1
        else:
            if _create_pending_entity(entity, document_id, db_url):
                stats["entities_pending"] += 1

    # Mark document as enriched
    with psycopg.connect(db_url) as conn:
        conn.execute(
            """
            UPDATE documents
            SET enriched_at = NOW(), enrichment_version = %s
            WHERE id = %s
            """,
            (config.version, document_id),
        )
        conn.commit()

    return stats


def get_unenriched_documents(
    db_url: str,
    source_type: str | None = None,
    project: str | None = None,
    query: str | None = None,
    path_pattern: str | None = None,
    enrichment_version: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Get documents that need enrichment.

    Args:
        db_url: Database URL
        source_type: Filter by source type
        project: Filter by project
        query: Semantic search query to filter documents
        path_pattern: SQL LIKE pattern to filter source_path (e.g., '%myrepo%')
        enrichment_version: Only include docs with older version (for re-enrichment)
        limit: Maximum documents to return

    Returns:
        List of document dicts with id, source_path, title, content, source_type
    """
    from ..local_embedder import embed_query

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        sql = """
            SELECT d.id, d.source_path, d.title, d.content, d.source_type, d.metadata
            FROM documents d
            WHERE (d.enriched_at IS NULL
        """
        params: list[Any] = []

        if enrichment_version is not None:
            sql += " OR d.enrichment_version < %s"
            params.append(enrichment_version)

        sql += ")"

        if source_type:
            sql += " AND d.source_type = %s"
            params.append(source_type)

        if project:
            sql += " AND d.metadata->>'project' = %s"
            params.append(project)

        if path_pattern:
            sql += " AND d.source_path LIKE %s"
            params.append(path_pattern)

        # Exclude already-derived documents (escape % for psycopg)
        sql += " AND d.source_path NOT LIKE '%%::todo/%%'"
        sql += " AND d.source_path NOT LIKE 'okb://entity/%%'"
        sql += " AND d.source_path NOT LIKE 'claude://%%'"

        if query:
            # Use semantic search to filter
            from pgvector.psycopg import register_vector

            register_vector(conn)
            embedding = embed_query(query)

            # Use GROUP BY to aggregate chunk similarities per document
            sql = """
                SELECT d.id, d.source_path, d.title, d.content, d.source_type, d.metadata,
                       MIN(c.embedding <=> %s::vector) as distance
                FROM documents d
                JOIN chunks c ON c.document_id = d.id
                WHERE (d.enriched_at IS NULL
            """
            params = [embedding]

            if enrichment_version is not None:
                sql += " OR d.enrichment_version < %s"
                params.append(enrichment_version)

            sql += ")"
            sql += " AND 1 - (c.embedding <=> %s::vector) > 0.3"
            params.append(embedding)

            if source_type:
                sql += " AND d.source_type = %s"
                params.append(source_type)

            if project:
                sql += " AND d.metadata->>'project' = %s"
                params.append(project)

            if path_pattern:
                sql += " AND d.source_path LIKE %s"
                params.append(path_pattern)

            sql += " AND d.source_path NOT LIKE '%%::todo/%%'"
            sql += " AND d.source_path NOT LIKE 'okb://entity/%%'"
            sql += " AND d.source_path NOT LIKE 'claude://%%'"

            sql += " GROUP BY d.id, d.source_path, d.title, d.content, d.source_type, d.metadata"
            sql += " ORDER BY distance"

        sql += f" LIMIT {limit}"

        results = conn.execute(sql, params).fetchall()
        return [dict(r) for r in results]


def list_pending_entities(
    db_url: str,
    entity_type: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List pending entity suggestions.

    Returns list of dicts with: id, entity_name, entity_type, aliases, description,
    mentions, confidence, source_path, source_title, created_at
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        sql = """
            SELECT
                pe.id, pe.entity_name, pe.entity_type, pe.aliases, pe.description,
                pe.mentions, pe.confidence, pe.created_at,
                d.source_path as source_path, d.title as source_title
            FROM pending_entities pe
            JOIN documents d ON d.id = pe.source_document_id
            WHERE pe.status = 'pending'
        """
        params: list[Any] = []

        if entity_type:
            sql += " AND pe.entity_type = %s"
            params.append(entity_type)

        sql += " ORDER BY pe.confidence DESC, pe.created_at DESC LIMIT %s"
        params.append(limit)

        results = conn.execute(sql, params).fetchall()
        return [dict(r) for r in results]


def approve_entity(db_url: str, pending_id: str, use_modal: bool = False) -> str | None:
    """Approve a pending entity, creating the entity document.

    Args:
        db_url: Database URL
        pending_id: ID of the pending entity to approve
        use_modal: If True, use Modal GPU for embedding; else local CPU

    Returns the entity source_path, or None if failed.
    """
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        # Get pending entity
        pe = conn.execute(
            """
            SELECT pe.*, d.id as source_doc_id
            FROM pending_entities pe
            JOIN documents d ON d.id = pe.source_document_id
            WHERE pe.id = %s AND pe.status = 'pending'
            """,
            (pending_id,),
        ).fetchone()

        if not pe:
            return None

        # Create entity from pending
        entity = ExtractedEntity(
            name=pe["entity_name"],
            entity_type=pe["entity_type"],
            aliases=pe["aliases"] or [],
            description=pe["description"],
            mentions=pe["mentions"] or [],
            confidence=pe["confidence"] or 0.8,
        )

        source_path = _create_entity_document(entity, pe["source_doc_id"], db_url, use_modal)

        if source_path:
            # Mark as approved
            conn.execute(
                """
                UPDATE pending_entities
                SET status = 'approved', reviewed_at = NOW()
                WHERE id = %s
                """,
                (pending_id,),
            )
            conn.commit()

        return source_path


def approve_entity_async(
    db_url: str, pending_id: str, use_modal: bool = False
) -> Future[str | None]:
    """Approve a pending entity asynchronously.

    The embedding and document creation happens in a background thread.
    Returns a Future that can be awaited or checked later.

    Args:
        db_url: Database URL
        pending_id: ID of the pending entity to approve
        use_modal: If True, use Modal GPU for embedding; else local CPU

    Returns:
        Future that resolves to the entity source_path, or None if failed.
    """
    executor = _get_executor()
    return executor.submit(approve_entity, db_url, pending_id, use_modal)


def reject_entity(db_url: str, pending_id: str) -> bool:
    """Reject a pending entity.

    Returns True if rejected, False if not found.
    """
    with psycopg.connect(db_url) as conn:
        result = conn.execute(
            """
            UPDATE pending_entities
            SET status = 'rejected', reviewed_at = NOW()
            WHERE id = %s AND status = 'pending'
            RETURNING id
            """,
            (pending_id,),
        ).fetchone()
        conn.commit()
        return result is not None
