# Local Knowledge Base (LKB)

A local-first semantic search system for personal documents with Claude Code integration via MCP.

## Installation

```bash
pip install local-kb
```

Or from source:
```bash
git clone https://github.com/yourusername/lkb
cd lkb
pip install -e .
```

## Quick Start

```bash
# 1. Start the database
lkb db start

# 2. (Optional) Deploy Modal embedder for faster batch ingestion
lkb modal deploy

# 3. Ingest your documents
lkb ingest ~/notes ~/docs

# 4. Configure Claude Code MCP (see below)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `lkb db start` | Start pgvector database container |
| `lkb db stop` | Stop database container |
| `lkb db status` | Show database status |
| `lkb db destroy` | Remove container and volume (destructive) |
| `lkb ingest <paths>` | Ingest documents into knowledge base |
| `lkb ingest <paths> --local` | Ingest using CPU embedding (no Modal) |
| `lkb serve` | Start MCP server (for Claude Code) |
| `lkb watch <paths>` | Watch directories for changes |
| `lkb config init` | Create default config file |
| `lkb config show` | Show current configuration |
| `lkb modal deploy` | Deploy GPU embedder to Modal |
| `lkb sync list` | List available API sources (plugins) |
| `lkb sync run <sources>` | Sync data from external APIs |
| `lkb sync status` | Show last sync times |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION (Burst GPU)                       │
│                                                                     │
│  Local Files → Contextual Chunking → Modal (GPU T4) → pgvector     │
│                                                                     │
│  ~/notes/project-x/api-design.md                                   │
│       ↓                                                             │
│  "Document: API Design Notes                                        │
│   Project: project-x                                                │
│   Section: Authentication                                           │
│   Content: Use JWT tokens with..."                                  │
│       ↓                                                             │
│  [0.23, -0.41, 0.87, ...]  → pgvector                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL (Always-on, Local)                   │
│                                                                     │
│  Claude Code → MCP Server → CPU Embedding → pgvector → Results     │
│                                                                     │
│  "How do I handle auth?"                                            │
│       ↓                                                             │
│  [0.19, -0.38, 0.91, ...]  (local CPU, ~300ms)                     │
│       ↓                                                             │
│  Cosine similarity search → Top 5 chunks with context              │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration

Configuration is loaded from `~/.config/lkb/config.yaml` (or `$XDG_CONFIG_HOME/lkb/config.yaml`).

Create default config:
```bash
lkb config init
```

Example config:
```yaml
database_url: postgresql://knowledge:localdev@localhost:5433/knowledge_base
docker:
  port: 5433
  container_name: lkb-pgvector
  volume_name: lkb-pgvector-data
  password: localdev
chunking:
  chunk_size: 512
  chunk_overlap: 64
```

Environment variables override config file settings:
- `KB_DATABASE_URL` - Database connection string
- `LKB_DOCKER_PORT` - Docker port mapping
- `LKB_CONTAINER_NAME` - Docker container name

## Claude Code MCP Config

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "lkb",
      "args": ["serve"]
    }
  }
}
```

## MCP Tools (Available in Claude Code)

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Semantic search with natural language queries |
| `keyword_search` | Exact keyword/symbol matching |
| `hybrid_search` | Combined semantic + keyword (RRF fusion) |
| `get_document` | Retrieve full document by path |
| `list_sources` | Show indexed document stats |
| `list_projects` | List known projects |
| `recent_documents` | Show recently indexed files |
| `save_knowledge` | Save knowledge from Claude for future reference |
| `delete_knowledge` | Delete a Claude-saved knowledge entry |
| `get_actionable_items` | Query tasks/events with structured filters |

## Contextual Chunking

Documents are chunked with context for better retrieval:

```
Document: Django Performance Notes
Project: student-app          ← inferred from path or frontmatter
Section: Query Optimization   ← extracted from markdown headers
Topics: django, performance   ← from frontmatter tags
Content: Use `select_related()` to avoid N+1 queries...
```

### Frontmatter Example

```markdown
---
tags: [django, postgresql, performance]
project: student-app
category: backend
---

# Query Optimization

Use `select_related()` for foreign keys...
```

## Cost Estimate

| Component | Local | Cloud Alternative |
|-----------|-------|-------------------|
| pgvector | $0 | ~$15-30/mo (CloudSQL) |
| MCP Server | $0 | ~$5/mo (small VM) |
| Modal embedding | ~$0.50-2/mo | N/A |
| **Total** | **~$1-2/mo** | **~$20-35/mo** |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check . && ruff format .
```

## Plugin System

LKB supports plugins for custom file parsers and API data sources (GitHub, Todoist, etc).

### Creating a Plugin

```python
# File parser plugin
from lkb.plugins import FileParser, Document

class EpubParser:
    extensions = ['.epub']
    source_type = 'epub'

    def can_parse(self, path): return path.suffix.lower() == '.epub'
    def parse(self, path, extra_metadata=None) -> Document: ...

# API source plugin
from lkb.plugins import APISource, SyncState, Document

class GitHubSource:
    name = 'github'
    source_type = 'github-issue'

    def configure(self, config): ...
    def fetch(self, state: SyncState | None) -> tuple[list[Document], SyncState]: ...
```

### Registering Plugins

In your plugin's `pyproject.toml`:
```toml
[project.entry-points."lkb.parsers"]
epub = "lkb_epub:EpubParser"

[project.entry-points."lkb.sources"]
github = "lkb_github:GitHubSource"
```

### Configuring API Sources

```yaml
# ~/.config/lkb/config.yaml
plugins:
  sources:
    github:
      enabled: true
      token: ${GITHUB_TOKEN}  # Resolved from environment
      repos: [owner/repo1, owner/repo2]
```

## License

MIT
