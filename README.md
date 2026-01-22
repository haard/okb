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
| `lkb serve` | Start MCP server (stdio, for Claude Code) |
| `lkb serve --http` | Start HTTP MCP server with token auth |
| `lkb watch <paths>` | Watch directories for changes |
| `lkb config init` | Create default config file |
| `lkb config show` | Show current configuration |
| `lkb modal deploy` | Deploy GPU embedder to Modal |
| `lkb token create` | Create API token for HTTP server |
| `lkb token list` | List tokens for a database |
| `lkb token revoke` | Revoke an API token |
| `lkb sync list` | List available API sources (plugins) |
| `lkb sync run <sources>` | Sync data from external APIs |
| `lkb sync status` | Show last sync times |
| `lkb rescan` | Check indexed files for changes, re-ingest stale |
| `lkb rescan --dry-run` | Show what would change without executing |
| `lkb rescan --delete` | Also remove documents for missing files |
| `lkb llm status` | Show LLM config and connectivity |
| `lkb llm deploy` | Deploy Modal LLM for open model inference |
| `lkb llm clear-cache` | Clear LLM response cache |

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
databases:
  personal:
    url: postgresql://knowledge:localdev@localhost:5433/personal_kb
    default: true    # Used when --db not specified (only one can be default)
    managed: true    # lkb manages via Docker
  work:
    url: postgresql://knowledge:localdev@localhost:5433/work_kb
    managed: true

docker:
  port: 5433
  container_name: lkb-pgvector

chunking:
  chunk_size: 512
  chunk_overlap: 64
```

Use `--db <name>` to target a specific database with any command.

Environment variables override config file settings:
- `KB_DATABASE_URL` - Database connection string
- `LKB_DOCKER_PORT` - Docker port mapping
- `LKB_CONTAINER_NAME` - Docker container name

### Project-Local Config

Override global config per-project with `.lkbconf.yaml` (searched from CWD upward):

```yaml
# .lkbconf.yaml
default_database: work  # Use 'work' db in this project

extensions:
  skip_directories:     # Extends global list
    - test_fixtures
```

Merge: scalars replace, lists extend, dicts deep-merge.

### LLM Integration (Optional)

Enable LLM-based document classification and filtering:

```yaml
llm:
  provider: claude          # "claude", "modal", or null (disabled)
  model: claude-haiku-4-5-20251001
  timeout: 30
  cache_responses: true
```

**Providers:**
| Provider | Setup | Cost |
|----------|-------|------|
| `claude` | `export ANTHROPIC_API_KEY=...` | ~$0.25/1M tokens |
| `modal` | `lkb llm deploy` | ~$0.02/min GPU |

For Modal (no API key needed):
```yaml
llm:
  provider: modal
  model: meta-llama/Llama-3.2-3B-Instruct
```

**Pre-ingest filtering** - skip low-value content during sync:
```yaml
plugins:
  sources:
    dropbox-paper:
      llm_filter:
        enabled: true
        prompt: "Skip meeting notes and drafts"
        action_on_skip: discard  # or "archive"
```

CLI commands:
```bash
lkb llm status              # Show config and connectivity
lkb llm deploy              # Deploy Modal LLM (for provider: modal)
lkb llm clear-cache         # Clear response cache
```

## Claude Code MCP Config

### stdio mode (default)

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

### HTTP mode (for remote/shared servers)

First, start the HTTP server and create a token:

```bash
# Create a token
lkb token create --db default -d "Claude Code"
# Output: lkb_default_rw_a1b2c3d4e5f6g7h8

# Start HTTP server
lkb serve --http --host 0.0.0.0 --port 8080
```

Then configure Claude Code to connect via SSE:

```json
{
  "mcpServers": {
    "knowledge-base": {
      "type": "sse",
      "url": "http://localhost:8080/sse",
      "headers": {
        "Authorization": "Bearer lkb_default_rw_a1b2c3d4e5f6g7h8"
      }
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
    dropbox-paper:
      enabled: true
      token: ${DROPBOX_TOKEN}
      folders: [/]            # Optional: filter to specific folders
```

## License

MIT
