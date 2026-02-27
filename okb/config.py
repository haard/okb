"""Shared configuration for the knowledge base."""

from __future__ import annotations

import os
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


class InsecureConfigError(Exception):
    """Raised when a config file has overly permissive permissions."""


def check_file_permissions(path: Path) -> None:
    """Check that a config file is not readable by group/other (must be mode 0600).

    Raises InsecureConfigError if permissions are too open.
    """
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        actual = stat.filemode(mode)
        print(
            f"WARNING: {path} has insecure permissions ({actual}).\n"
            f"Config files may contain secrets and must not be readable by others.\n"
            f"Fix with: chmod 600 {path}",
            file=sys.stderr,
        )
        raise InsecureConfigError(
            f"{path} has insecure permissions ({actual}). Run: chmod 600 {path}"
        )


def resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} references in config values.

    Supports:
    - ${VAR} - required, raises if not set
    - ${VAR:-default} - optional with default value

    Args:
        value: Config value (string, dict, or list)

    Returns:
        Value with env vars resolved
    """
    if isinstance(value, str):
        # Pattern: ${VAR} or ${VAR:-default}
        pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ValueError(f"Environment variable ${var_name} not set and no default provided")

        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
    return value


@dataclass
class DatabaseConfig:
    """Configuration for a single database."""

    name: str
    url: str
    managed: bool = True  # Whether okb manages this (Docker) or external
    default: bool = False
    description: str | None = None  # Human-readable description for LLM context
    topics: list[str] | None = None  # Topic keywords to help LLM route queries
    sources: dict[str, dict] | None = None  # Per-database source config (overrides global)

    @property
    def database_name(self) -> str:
        """Extract database name from URL."""
        parsed = urlparse(self.url)
        return parsed.path.lstrip("/") or self.name


@dataclass
class ServerConfig:
    """Configuration for a remote MCP server."""

    name: str
    url: str
    token: str | None = None
    default: bool = False


def get_config_dir() -> Path:
    """Get the config directory, respecting XDG_CONFIG_HOME."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "okb"
    return Path.home() / ".config" / "okb"


def get_config_path() -> Path:
    """Get the path to the config file."""
    return get_config_dir() / "config.yaml"


def get_data_dir() -> Path:
    """Get the data directory, respecting XDG_DATA_HOME."""
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "okb"
    return Path.home() / ".local" / "share" / "okb"


def get_snapshots_dir(database_name: str) -> Path:
    """Get snapshots directory for a database."""
    return get_data_dir() / "snapshots" / database_name


def load_config_file() -> dict[str, Any]:
    """Load configuration from config file if it exists.

    Raises InsecureConfigError if file permissions are too open.
    """
    config_path = get_config_path()
    if config_path.exists():
        check_file_permissions(config_path)
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def find_local_config(start_path: Path | None = None) -> Path | None:
    """Find .okbconf.yaml by walking up from start_path (default: CWD)."""
    path = (start_path or Path.cwd()).resolve()
    while path != path.parent:
        local_config = path / ".okbconf.yaml"
        if local_config.exists():
            return local_config
        path = path.parent
    return None


def load_local_config() -> dict[str, Any]:
    """Load local config overlay if present.

    Raises InsecureConfigError if file permissions are too open.
    """
    local_path = find_local_config()
    if local_path:
        check_file_permissions(local_path)
        with open(local_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_configs(base: dict, overlay: dict, path: str = "") -> dict:
    """Merge overlay config into base. Lists extend, dicts deep-merge, scalars replace.

    Raises ValueError if types don't match (e.g., list vs dict).
    """
    result = dict(base)
    for key, value in overlay.items():
        key_path = f"{path}.{key}" if path else key
        if key in result:
            base_val = result[key]
            # Check for type mismatches between collections
            if isinstance(value, list) and not isinstance(base_val, list):
                raise ValueError(
                    f"Config type mismatch at '{key_path}': "
                    f"global config has {type(base_val)}, local config has list"
                )
            if isinstance(value, dict) and not isinstance(base_val, dict):
                raise ValueError(
                    f"Config type mismatch at '{key_path}': "
                    f"global config has {type(base_val)}, local config has dict"
                )
            # Merge by type
            if isinstance(value, list):
                result[key] = base_val + value  # Extend lists
            elif isinstance(value, dict):
                result[key] = merge_configs(base_val, value, key_path)  # Deep merge dicts
            else:
                result[key] = value  # Replace scalar
        else:
            result[key] = value
    return result


# Default configuration values
DEFAULTS = {
    "databases": {
        "default": {
            "url": "postgresql://knowledge:localdev@localhost:5433/knowledge_base",
            "default": True,
            "managed": True,
        },
    },
    "docker": {
        "port": 5433,
        "container_name": "okb-pgvector",
        "volume_name": "okb-pgvector-data",
        "password": "localdev",
    },
    "http": {
        "host": "127.0.0.1",
        "port": 8080,
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
        "documents": [".md", ".txt", ".markdown", ".org", ".pdf", ".docx"],
        "code": [
            ".py",
            ".rb",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".sql",
            ".sh",
            ".bash",
            ".fish",
            ".yaml",
            ".yml",
            ".toml",
            ".json",
            ".html",
            ".css",
            ".scss",
            ".go",
            ".rs",
            ".java",
            ".kt",
            ".c",
            ".cpp",
            ".h",
        ],
        "skip_directories": [
            ".git",
            ".hg",
            ".svn",
            "vault",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            ".next",
            ".nuxt",
            "lib",
            "libs",
            "vendor",
            "third_party",
            "third-party",
            "external",
            "bower_components",
        ],
    },
    "security": {
        # Sensitive files - blocked by default
        "block_patterns": [
            "id_rsa",
            "id_ed25519",
            "id_ecdsa",
            "id_dsa",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            ".env",
            ".env.*",
            "*credentials*",
            "*secret*",
            ".netrc",
            ".pgpass",
            ".my.cnf",
            "*_history",
            ".bash_history",
            ".zsh_history",
        ],
        # Low-value files - skipped for usefulness not security
        "skip_patterns": [
            "*.min.js",
            "*.min.css",
            "*.bundle.js",
            "*.chunk.js",
            "*.map",
            "package-lock.json",
            "yarn.lock",
            "uv.lock",
            "Cargo.lock",
            "poetry.lock",
            "*.pyc",
            "*.pyo",
            "*.tmp",
            "*.tmp.*",
            ".#*",
            "*~",  # Temp/backup files
        ],
        "scan_content": True,
        "max_line_length_for_minified": 1000,
    },
    "plugins": {
        # API sources configuration
        # Example:
        # sources:
        #   github:
        #     enabled: true
        #     token: ${GITHUB_TOKEN}
        #     repos: [owner/repo1, owner/repo2]
        "sources": {},
    },
    "llm": {
        # LLM provider configuration
        # provider: None = disabled, "claude" = Anthropic API, "modal" = Modal GPU
        "provider": None,
        "model": "claude-haiku-4-5-20251001",
        "timeout": 30,
        "cache_responses": True,
        # Bedrock settings (when use_bedrock is True)
        "use_bedrock": False,
        "aws_region": "us-west-2",
        # Modal settings (when provider is "modal")
        "modal_gpu": "L4",  # GPU type: T4, L4, A10G, A100, etc.
    },
}


@dataclass
class Config:
    """Knowledge base configuration."""

    # Multiple databases support
    databases: dict[str, DatabaseConfig] = field(default_factory=dict)
    default_database: str | None = None

    # Local config overlay path (set in __post_init__ if found)
    local_config_path: Path | None = None

    # Docker
    docker_port: int = 5433
    docker_container_name: str = "okb-pgvector"
    docker_volume_name: str = "okb-pgvector-data"
    docker_password: str = "localdev"

    # HTTP server
    http_host: str = "127.0.0.1"
    http_port: int = 8080

    # Remote servers (for okb client CLI)
    servers: dict[str, ServerConfig] = field(default_factory=dict)
    default_server: str | None = None

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

    # Security settings (loaded from config in __post_init__)
    block_patterns: list[str] = field(default_factory=list)
    skip_patterns: list[str] = field(default_factory=list)
    scan_content: bool = True
    max_line_length_for_minified: int = 1000

    # Plugin settings (loaded from config in __post_init__)
    plugin_sources: dict[str, dict] = field(default_factory=dict)

    # LLM settings (loaded from config in __post_init__)
    llm_provider: str | None = None
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_timeout: int = 30
    llm_cache_responses: bool = True
    llm_use_bedrock: bool = False
    llm_aws_region: str = "us-west-2"
    llm_modal_gpu: str = "L4"

    def __post_init__(self):
        """Load configuration from file and environment."""
        file_config = load_config_file()

        # Load and merge local config overlay (.okbconf.yaml)
        local_path = find_local_config()
        local_default_db: str | None = None
        local_default_server: str | None = None
        if local_path:
            self.local_config_path = local_path
            local_config = load_local_config()
            file_config = merge_configs(file_config, local_config)

            # Save local config's defaults to apply after database/server loading
            if "default_database" in local_config:
                local_default_db = local_config["default_database"]
            if "default_server" in local_config:
                local_default_server = local_config["default_server"]

        # Merge extension/security lists with defaults so local config extends defaults
        # (not just global config file which may be empty)
        list_fields_to_extend = [
            ("extensions", ["skip_directories"]),
            ("security", ["block_patterns", "skip_patterns"]),
        ]
        for section, keys in list_fields_to_extend:
            if section in file_config:
                for key in keys:
                    if key in file_config[section]:
                        # Prepend defaults to user's list (user values extend defaults)
                        default_list = DEFAULTS[section][key]
                        user_list = file_config[section][key]
                        # Deduplicate while preserving order
                        seen = set()
                        merged = []
                        for item in default_list + user_list:
                            if item not in seen:
                                seen.add(item)
                                merged.append(item)
                        file_config[section][key] = merged

        # Load databases: new multi-db format or legacy single database_url
        if "databases" in file_config:
            default_dbs = []
            for name, db_cfg in file_config["databases"].items():
                self.databases[name] = DatabaseConfig(
                    name=name,
                    url=db_cfg["url"],
                    managed=db_cfg.get("managed", True),
                    default=db_cfg.get("default", False),
                    description=db_cfg.get("description"),
                    topics=db_cfg.get("topics"),
                    sources=db_cfg.get("sources"),
                )
                if db_cfg.get("default"):
                    default_dbs.append(name)
                    self.default_database = name
            # Validate only one default
            if len(default_dbs) > 1:
                raise ValueError(
                    f"Multiple databases marked as default: {default_dbs}. "
                    "Only one database can have 'default: true'."
                )
            # If no default was marked, use first database
            if not self.default_database and self.databases:
                first_name = next(iter(self.databases))
                self.databases[first_name].default = True
                self.default_database = first_name
        else:
            # Legacy: single database_url (env > file > default)
            legacy_url = os.environ.get(
                "OKB_DATABASE_URL",
                file_config.get("database_url", DEFAULTS["databases"]["default"]["url"]),
            )
            self.databases["default"] = DatabaseConfig(
                name="default",
                url=legacy_url,
                managed=True,
                default=True,
            )
            self.default_database = "default"

        # Apply local config's default_database override (takes precedence over global)
        if local_default_db:
            self.default_database = local_default_db

        # Docker settings
        docker_cfg = file_config.get("docker", {})
        self.docker_port = int(
            os.environ.get(
                "OKB_DOCKER_PORT",
                docker_cfg.get("port", DEFAULTS["docker"]["port"]),
            )
        )
        self.docker_container_name = os.environ.get(
            "OKB_CONTAINER_NAME",
            docker_cfg.get("container_name", DEFAULTS["docker"]["container_name"]),
        )
        self.docker_volume_name = os.environ.get(
            "OKB_VOLUME_NAME",
            docker_cfg.get("volume_name", DEFAULTS["docker"]["volume_name"]),
        )
        self.docker_password = os.environ.get(
            "OKB_DB_PASSWORD",
            docker_cfg.get("password", DEFAULTS["docker"]["password"]),
        )

        # HTTP server settings
        http_cfg = file_config.get("http", {})
        self.http_host = os.environ.get(
            "OKB_HTTP_HOST",
            http_cfg.get("host", DEFAULTS["http"]["host"]),
        )
        self.http_port = int(
            os.environ.get(
                "OKB_HTTP_PORT",
                http_cfg.get("port", DEFAULTS["http"]["port"]),
            )
        )

        # Remote server settings (for okb client CLI)
        if "servers" in file_config:
            default_servers = []
            for sname, srv_cfg in file_config["servers"].items():
                token_raw = srv_cfg.get("token")
                token_val = None
                if token_raw:
                    try:
                        token_val = resolve_env_vars(token_raw)
                    except ValueError:
                        token_val = token_raw
                self.servers[sname] = ServerConfig(
                    name=sname,
                    url=srv_cfg["url"],
                    token=token_val,
                    default=srv_cfg.get("default", False),
                )
                if srv_cfg.get("default"):
                    default_servers.append(sname)
                    self.default_server = sname
            if len(default_servers) > 1:
                raise ValueError(
                    f"Multiple servers marked as default: {default_servers}. "
                    "Only one server can have 'default: true'."
                )
            if not self.default_server and self.servers:
                first_name = next(iter(self.servers))
                self.servers[first_name].default = True
                self.default_server = first_name

        # Apply local config's default_server override
        if local_default_server:
            self.default_server = local_default_server

        # Env var overrides apply to the default server
        env_url = os.environ.get("OKB_SERVER_URL")
        env_token_raw = os.environ.get("OKB_TOKEN")
        if env_url or env_token_raw:
            if self.default_server and self.default_server in self.servers:
                sc = self.servers[self.default_server]
                if env_url:
                    sc.url = env_url
                if env_token_raw:
                    try:
                        sc.token = resolve_env_vars(env_token_raw)
                    except ValueError:
                        sc.token = env_token_raw
            elif env_url:
                # No servers configured yet — create one from env vars
                token_val = None
                if env_token_raw:
                    try:
                        token_val = resolve_env_vars(env_token_raw)
                    except ValueError:
                        token_val = env_token_raw
                self.servers["default"] = ServerConfig(
                    name="default",
                    url=env_url,
                    token=token_val,
                    default=True,
                )
                self.default_server = "default"

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
        self.code_extensions = frozenset(ext_cfg.get("code", DEFAULTS["extensions"]["code"]))
        self.skip_directories = frozenset(
            ext_cfg.get("skip_directories", DEFAULTS["extensions"]["skip_directories"])
        )

        # Security settings
        security_cfg = file_config.get("security", {})
        self.block_patterns = security_cfg.get(
            "block_patterns", DEFAULTS["security"]["block_patterns"]
        )
        self.skip_patterns = security_cfg.get(
            "skip_patterns", DEFAULTS["security"]["skip_patterns"]
        )
        self.scan_content = security_cfg.get("scan_content", DEFAULTS["security"]["scan_content"])
        self.max_line_length_for_minified = security_cfg.get(
            "max_line_length_for_minified", DEFAULTS["security"]["max_line_length_for_minified"]
        )

        # Plugin settings - resolve env vars in source configs
        plugins_cfg = file_config.get("plugins", {})
        self.plugin_sources = plugins_cfg.get("sources", {})

        # LLM settings
        llm_cfg = file_config.get("llm", {})
        self.llm_provider = os.environ.get(
            "OKB_LLM_PROVIDER",
            llm_cfg.get("provider", DEFAULTS["llm"]["provider"]),
        )
        self.llm_model = os.environ.get(
            "OKB_LLM_MODEL",
            llm_cfg.get("model", DEFAULTS["llm"]["model"]),
        )
        self.llm_timeout = int(
            os.environ.get(
                "OKB_LLM_TIMEOUT",
                llm_cfg.get("timeout", DEFAULTS["llm"]["timeout"]),
            )
        )
        self.llm_cache_responses = llm_cfg.get(
            "cache_responses", DEFAULTS["llm"]["cache_responses"]
        )
        self.llm_use_bedrock = llm_cfg.get("use_bedrock", DEFAULTS["llm"]["use_bedrock"])
        self.llm_aws_region = llm_cfg.get("aws_region", DEFAULTS["llm"]["aws_region"])
        self.llm_modal_gpu = os.environ.get(
            "OKB_MODAL_GPU",
            llm_cfg.get("modal_gpu", DEFAULTS["llm"]["modal_gpu"]),
        )

    def get_database(self, name: str | None = None) -> DatabaseConfig:
        """Get database config by name, or default if None."""
        if name is None:
            name = self.default_database
        if name is None:
            raise ValueError("No database specified and no default configured")
        if name not in self.databases:
            raise ValueError(f"Unknown database: {name}. Available: {list(self.databases.keys())}")
        return self.databases[name]

    def get_server(self, name: str | None = None) -> ServerConfig:
        """Get server config by name, or default if None."""
        if name is None:
            name = self.default_server
        if name is None:
            raise ValueError(
                "No server configured. Add servers: to config or set OKB_SERVER_URL."
            )
        if name not in self.servers:
            raise ValueError(
                f"Unknown server: {name}. Available: {list(self.servers.keys())}"
            )
        return self.servers[name]

    @property
    def db_url(self) -> str:
        """Backward compat: return default database URL."""
        return self.get_database().url

    @property
    def all_extensions(self) -> frozenset[str]:
        return self.document_extensions | self.code_extensions

    def should_skip_path(self, path: Path) -> bool:
        """Check if a path should be skipped during collection."""
        return any(part.startswith(".") or part in self.skip_directories for part in path.parts)

    def _get_sources_for_db(self, db_name: str | None) -> dict[str, dict]:
        """Get the effective source config dict for a database.

        If the database has per-db sources defined for a given source name,
        those replace the global config entirely (no deep merge).
        Sources not defined at db level fall back to global plugin_sources.
        """
        if db_name:
            db_cfg = self.databases.get(db_name)
            if db_cfg and db_cfg.sources:
                # Start with global, then overlay per-db sources (full replacement per source)
                merged = dict(self.plugin_sources)
                merged.update(db_cfg.sources)
                return merged
        return self.plugin_sources

    def get_source_config(self, source_name: str, db_name: str | None = None) -> dict | None:
        """Get resolved config for a plugin source.

        If db_name is given and the database defines config for this source,
        that config replaces the global config entirely. Otherwise falls back
        to global plugin_sources.

        Resolves ${ENV_VAR} references in the config values.
        Returns None if source not configured or disabled.
        """
        sources = self._get_sources_for_db(db_name)
        source_cfg = sources.get(source_name)
        if source_cfg is None:
            return None
        if not source_cfg.get("enabled", True):
            return None
        try:
            return resolve_env_vars(source_cfg)
        except ValueError as e:
            raise ValueError(f"Error resolving config for source '{source_name}': {e}") from e

    def list_enabled_sources(self, db_name: str | None = None) -> list[str]:
        """List all enabled plugin sources for a database (or global)."""
        sources = self._get_sources_for_db(db_name)
        return [name for name, cfg in sources.items() if cfg.get("enabled", True)]

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for display."""
        databases_dict = {}
        for name, db_cfg in self.databases.items():
            db_dict: dict[str, Any] = {
                "url": db_cfg.url,
                "managed": db_cfg.managed,
                "default": db_cfg.default,
            }
            if db_cfg.description:
                db_dict["description"] = db_cfg.description
            if db_cfg.topics:
                db_dict["topics"] = db_cfg.topics
            if db_cfg.sources:
                db_dict["sources"] = {
                    sname: {**scfg, "token": "***" if "token" in scfg else None}
                    for sname, scfg in db_cfg.sources.items()
                }
            databases_dict[name] = db_dict

        result: dict[str, Any] = {}
        if self.local_config_path:
            result["local_config"] = str(self.local_config_path)
        result["databases"] = databases_dict
        if self.servers:
            result["servers"] = {
                sname: {
                    "url": sc.url,
                    "token": "***" if sc.token else None,
                    "default": sc.default,
                }
                for sname, sc in self.servers.items()
            }
        return {
            **result,
            "docker": {
                "port": self.docker_port,
                "container_name": self.docker_container_name,
                "volume_name": self.docker_volume_name,
                "password": "***" if self.docker_password else None,
            },
            "http": {
                "host": self.http_host,
                "port": self.http_port,
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
            "security": {
                "block_patterns": self.block_patterns,
                "skip_patterns": self.skip_patterns,
                "scan_content": self.scan_content,
                "max_line_length_for_minified": self.max_line_length_for_minified,
            },
            "plugins": {
                "sources": {
                    name: {**cfg, "token": "***" if "token" in cfg else None}
                    for name, cfg in self.plugin_sources.items()
                },
            },
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "timeout": self.llm_timeout,
                "cache_responses": self.llm_cache_responses,
                "use_bedrock": self.llm_use_bedrock,
                "aws_region": self.llm_aws_region,
                "modal_gpu": self.llm_modal_gpu,
            },
        }


def get_default_config_yaml() -> str:
    """Get the default config template with all values commented out."""
    return """\
# OKB configuration
# Uncomment and modify values to override defaults.
# Environment variables take precedence over this file.

# --- Databases ---
# Multiple knowledge bases can be configured. Only one can be default.
# databases:
#   default:
#     url: postgresql://knowledge:localdev@localhost:5433/knowledge_base
#     default: true
#     managed: true              # okb manages via Docker container
#     description: "Personal notes and documents"
#     topics: [notes, journal]
#   work:
#     url: postgresql://knowledge:localdev@localhost:5433/work_kb
#     managed: true
#     description: "Work projects and documentation"
#     topics: [work, projects]

# --- Docker (pgvector container) ---
# docker:
#   port: 5433
#   container_name: okb-pgvector
#   volume_name: okb-pgvector-data
#   password: localdev

# --- HTTP server ---
# http:
#   host: 127.0.0.1
#   port: 8080

# --- Remote servers (for okb client CLI) ---
# Env vars OKB_SERVER_URL / OKB_TOKEN override the default server.
# servers:
#   personal:
#     url: http://localhost:8080/mcp
#     token: ${OKB_PERSONAL_TOKEN}
#     default: true
#   work:
#     url: http://work-host:8080/mcp
#     token: ${OKB_WORK_TOKEN}

# --- Embedding model ---
# embedding:
#   model_name: nomic-ai/nomic-embed-text-v1.5
#   dimension: 768

# --- Chunking ---
# chunking:
#   chunk_size: 512              # tokens per chunk
#   chunk_overlap: 64            # overlap between chunks
#   chars_per_token: 4

# --- Search ---
# search:
#   default_limit: 5
#   max_limit: 20
#   min_similarity: 0.3

# --- File extensions ---
# Local config (.okbconf.yaml) extends these lists rather than replacing them.
# extensions:
#   documents: [.md, .txt, .markdown, .org, .pdf, .docx]
#   code: [.py, .rb, .js, .ts, .jsx, .tsx, .sql, .sh, .bash, .fish,
#          .yaml, .yml, .toml, .json, .html, .css, .scss,
#          .go, .rs, .java, .kt, .c, .cpp, .h]
#   skip_directories: [.git, node_modules, __pycache__, .venv, venv,
#                      dist, build, vendor]

# --- Security ---
# security:
#   scan_content: true
#   max_line_length_for_minified: 1000
#   block_patterns:              # sensitive files, always skipped
#     - "*.pem"
#     - "*.key"
#     - ".env"
#     - ".env.*"
#   skip_patterns:               # low-value files
#     - "*.min.js"
#     - "package-lock.json"
#     - "poetry.lock"

# --- Plugins (API sources) ---
# Token values support ${ENV_VAR} and ${ENV_VAR:-default} syntax.
# plugins:
#   sources:
#     github:
#       enabled: true
#       token: ${GITHUB_TOKEN}
#     todoist:
#       enabled: true
#       token: ${TODOIST_TOKEN}
#       include_completed: false
#       completed_days: 30
#     dropbox-paper:
#       enabled: true
#       app_key: ${DROPBOX_APP_KEY}
#       app_secret: ${DROPBOX_APP_SECRET}
#       refresh_token: ${DROPBOX_REFRESH_TOKEN}
#       folders: [/]

# --- LLM (for synthesis and classification) ---
# llm:
#   provider: claude              # claude, modal, or null (disabled)
#   model: claude-haiku-4-5-20251001
#   timeout: 30
#   cache_responses: true
#   use_bedrock: false
#   aws_region: us-west-2
"""


class _LazyConfig:
    """Lazy proxy that defers Config() instantiation until first attribute access."""

    def __init__(self):
        object.__setattr__(self, "_instance", None)

    def _get(self) -> Config:
        inst = object.__getattribute__(self, "_instance")
        if inst is None:
            inst = Config()
            object.__setattr__(self, "_instance", inst)
        return inst

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __setattr__(self, name, value):
        setattr(self._get(), name, value)


# Global config instance (lazily initialized on first access)
config: Config = _LazyConfig()  # type: ignore[assignment]
