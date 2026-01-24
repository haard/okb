"""
Modal-based GPU embedding service.

Provides on-demand GPU access for batch embedding generation.
Costs approximately $0.02 per 1000 chunks on T4 GPU.

Usage:
    modal deploy modal_embedder.py

Then call from Python:
    embedder = modal.Cls.from_name("knowledge-embedder", "Embedder")()
    embeddings = embedder.embed_batch.remote(texts)
"""

import modal

app = modal.App("knowledge-embedder")

# Container image with all dependencies
embedder_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "einops>=0.7.0",  # Required by nomic
)


@app.cls(
    image=embedder_image,
    gpu="T4",  # Cheapest option, sufficient for embedding
    timeout=600,
    scaledown_window=300,  # Keep warm for 5 min
    retries=2,
)
class Embedder:
    """GPU-accelerated embedding generator using nomic-embed-text."""

    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        from sentence_transformers import SentenceTransformer
        import torch

        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        print(f"Model loaded on {device}")

    @modal.method()
    def embed_batch(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int = 16,
    ) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text chunks to embed
            is_query: If True, use query prefix; otherwise document prefix
            batch_size: Processing batch size

        Returns:
            List of embedding vectors (768 dimensions)
        """
        # Nomic model requires task-specific prefixes
        prefix = "search_query: " if is_query else "search_document: "
        prefixed = [f"{prefix}{t}" for t in texts]

        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )

        return embeddings.tolist()

    @modal.method()
    def embed_single(self, text: str, is_query: bool = False) -> list[float]:
        """Embed a single text (convenience method)."""
        prefix = "search_query: " if is_query else "search_document: "
        embedding = self.model.encode(
            f"{prefix}{text}",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()


# For testing
@app.local_entrypoint()
def test():
    """Test the embedder."""
    embedder = Embedder()

    test_texts = [
        "Django ORM query optimization using select_related and prefetch_related",
        "PostgreSQL VACUUM and ANALYZE for table maintenance",
        "Kubernetes pod scheduling with node affinity rules",
    ]

    print(f"Embedding {len(test_texts)} texts...")
    embeddings = embedder.embed_batch.remote(test_texts)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Test similarity
    import numpy as np

    emb = np.array(embeddings)
    similarity = np.dot(emb, emb.T)
    print(f"\nSimilarity matrix:\n{similarity}")
