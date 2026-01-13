"""Shared configuration for the knowledge base."""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Knowledge base configuration."""

    # Database
    db_url: str = field(
        default_factory=lambda: os.environ.get(
            "KB_DATABASE_URL",
            "postgresql://knowledge:localdev@localhost:5433/knowledge_base",
        )
    )

    # Embedding model
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768

    # Chunking
    chunk_size: int = 512  # tokens (approx)
    chunk_overlap: int = 64  # tokens (approx)
    chars_per_token: int = 4  # rough approximation

    # Search defaults
    default_limit: int = 5
    max_limit: int = 20
    min_similarity: float = 0.3

    # File types
    markdown_extensions: frozenset[str] = frozenset({".md", ".txt", ".markdown"})
    code_extensions: frozenset[str] = frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".sql", ".sh", ".bash",
        ".yaml", ".yml",  ".toml",
        ".html", ".css", ".scss",
        ".go", ".rs", ".java", ".kt", ".fish"
    })
    skip_directories: frozenset[str] = frozenset({
        ".git", ".hg", ".svn", "vault",
        "node_modules", "__pycache__", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "dist", "build", ".next", ".nuxt",
    })

    @property
    def all_extensions(self) -> frozenset[str]:
        return self.markdown_extensions | self.code_extensions

    def should_skip_path(self, path: Path) -> bool:
        """Check if a path should be skipped during collection."""
        return any(
            part.startswith(".") or part in self.skip_directories
            for part in path.parts
        )


# Global config instance
config = Config()
