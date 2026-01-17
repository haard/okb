"""
Local CPU-based embedder for query-time use.

Avoids Modal cold starts for interactive queries.
Latency: ~200-500ms per query on modern CPU
Memory: ~1.5 GB for model

The same nomic-embed-text model is used for consistency.
"""

from functools import lru_cache

from .config import config


@lru_cache(maxsize=1)
def get_model():
    """
    Load embedding model once, cache in memory.

    First call takes ~10-30 seconds to download/load.
    Subsequent calls return cached model instantly.

    Auto-detects GPU (CUDA/MPS) and uses it if available.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {config.model_name}")
    model = SentenceTransformer(
        config.model_name,
        trust_remote_code=True,
    )
    # Report actual device being used
    device = str(model.device)
    print(f"Model loaded on {device}")
    return model


def embed_query(text: str) -> list[float]:
    """
    Generate embedding for a search query.

    Uses "search_query: " prefix as required by nomic model.
    """
    model = get_model()
    embedding = model.encode(
        f"search_query: {text}",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def embed_document(text: str) -> list[float]:
    """
    Generate embedding for a document chunk.

    Uses "search_document: " prefix as required by nomic model.
    Prefer Modal for batch document embedding.
    """
    model = get_model()
    embedding = model.encode(
        f"search_document: {text}",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def warmup():
    """
    Pre-load model to avoid first-query latency.

    Call this at server startup.
    """
    get_model()


if __name__ == "__main__":
    # Quick test
    warmup()

    test_query = "How do I optimize Django database queries?"
    embedding = embed_query(test_query)
    print(f"Query: {test_query}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
