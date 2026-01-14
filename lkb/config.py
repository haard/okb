"""Shared configuration for the knowledge base."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def get_config_dir() -> Path:
    """Get the config directory, respecting XDG_CONFIG_HOME."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "lkb"
    return Path.home() / ".config" / "lkb"


def get_config_path() -> Path:
    """Get the path to the config file."""
    return get_config_dir() / "config.yaml"


def load_config_file() -> dict[str, Any]:
    """Load configuration from config file if it exists."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# Default configuration values
DEFAULTS = {
    "database_url": "postgresql://knowledge:localdev@localhost:5433/knowledge_base",
    "docker": {
        "port": 5433,
        "container_name": "lkb-pgvector",
        "volume_name": "lkb-pgvector-data",
        "password": "localdev",
    },
    "embedding": {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "dimension": 768,
    },
    "chunking": {
        "chunk_size": 512,
        "chunk_overlap": 64,
        "chars_per_token": 4,
    },
    "search": {
        "default_limit": 5,
        "max_limit": 20,
        "min_similarity": 0.3,
    },
    "extensions": {
        "documents": [".md", ".txt", ".markdown", ".org"],
        "code": [
            ".py", ".rb", ".js", ".ts", ".jsx", ".tsx",
            ".sql", ".sh", ".bash", ".fish",
            ".yaml", ".yml", ".toml", ".json",
            ".html", ".css", ".scss",
            ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".h",
        ],
        "skip_directories": [
            ".git", ".hg", ".svn", "vault",
            "node_modules", "__pycache__", ".venv", "venv",
            ".mypy_cache", ".pytest_cache", ".ruff_cache",
            "dist", "build", ".next", ".nuxt",
        ],
    },
}


@dataclass
class Config:
    """Knowledge base configuration."""

    # Database
    db_url: str = field(default="")

    # Docker
    docker_port: int = 5433
    docker_container_name: str = "lkb-pgvector"
    docker_volume_name: str = "lkb-pgvector-data"
    docker_password: str = "localdev"

    # Embedding model
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    chars_per_token: int = 4

    # Search defaults
    default_limit: int = 5
    max_limit: int = 20
    min_similarity: float = 0.3

    # File types (loaded from config in __post_init__)
    document_extensions: frozenset[str] = field(default_factory=frozenset)
    code_extensions: frozenset[str] = field(default_factory=frozenset)
    skip_directories: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        """Load configuration from file and environment."""
        file_config = load_config_file()

        # Database URL: env > file > default
        if not self.db_url:
            self.db_url = os.environ.get(
                "KB_DATABASE_URL",
                file_config.get("database_url", DEFAULTS["database_url"]),
            )

        # Docker settings
        docker_cfg = file_config.get("docker", {})
        self.docker_port = int(
            os.environ.get(
                "LKB_DOCKER_PORT",
                docker_cfg.get("port", DEFAULTS["docker"]["port"]),
            )
        )
        self.docker_container_name = os.environ.get(
            "LKB_CONTAINER_NAME",
            docker_cfg.get("container_name", DEFAULTS["docker"]["container_name"]),
        )
        self.docker_volume_name = os.environ.get(
            "LKB_VOLUME_NAME",
            docker_cfg.get("volume_name", DEFAULTS["docker"]["volume_name"]),
        )
        self.docker_password = os.environ.get(
            "LKB_DB_PASSWORD",
            docker_cfg.get("password", DEFAULTS["docker"]["password"]),
        )

        # Embedding settings
        embedding_cfg = file_config.get("embedding", {})
        self.model_name = embedding_cfg.get("model_name", DEFAULTS["embedding"]["model_name"])
        self.embedding_dim = embedding_cfg.get("dimension", DEFAULTS["embedding"]["dimension"])

        # Chunking settings
        chunking_cfg = file_config.get("chunking", {})
        self.chunk_size = chunking_cfg.get("chunk_size", DEFAULTS["chunking"]["chunk_size"])
        self.chunk_overlap = chunking_cfg.get(
            "chunk_overlap", DEFAULTS["chunking"]["chunk_overlap"]
        )
        self.chars_per_token = chunking_cfg.get(
            "chars_per_token", DEFAULTS["chunking"]["chars_per_token"]
        )

        # Search settings
        search_cfg = file_config.get("search", {})
        self.default_limit = search_cfg.get("default_limit", DEFAULTS["search"]["default_limit"])
        self.max_limit = search_cfg.get("max_limit", DEFAULTS["search"]["max_limit"])
        self.min_similarity = search_cfg.get("min_similarity", DEFAULTS["search"]["min_similarity"])

        # Extension settings
        ext_cfg = file_config.get("extensions", {})
        self.document_extensions = frozenset(
            ext_cfg.get("documents", DEFAULTS["extensions"]["documents"])
        )
        self.code_extensions = frozenset(
            ext_cfg.get("code", DEFAULTS["extensions"]["code"])
        )
        self.skip_directories = frozenset(
            ext_cfg.get("skip_directories", DEFAULTS["extensions"]["skip_directories"])
        )

    @property
    def all_extensions(self) -> frozenset[str]:
        return self.document_extensions | self.code_extensions

    def should_skip_path(self, path: Path) -> bool:
        """Check if a path should be skipped during collection."""
        return any(part.startswith(".") or part in self.skip_directories for part in path.parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for display."""
        return {
            "database_url": self.db_url,
            "docker": {
                "port": self.docker_port,
                "container_name": self.docker_container_name,
                "volume_name": self.docker_volume_name,
                "password": "***" if self.docker_password else None,
            },
            "embedding": {
                "model_name": self.model_name,
                "dimension": self.embedding_dim,
            },
            "chunking": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chars_per_token": self.chars_per_token,
            },
            "search": {
                "default_limit": self.default_limit,
                "max_limit": self.max_limit,
                "min_similarity": self.min_similarity,
            },
            "extensions": {
                "documents": sorted(self.document_extensions),
                "code": sorted(self.code_extensions),
                "skip_directories": sorted(self.skip_directories),
            },
        }


def get_default_config_yaml() -> str:
    """Get the default config as YAML string."""
    return yaml.dump(DEFAULTS, default_flow_style=False, sort_keys=False)


# Global config instance
config = Config()
