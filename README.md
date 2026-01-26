# Owned Knowledge Base (OKB)

A local-first semantic search system for personal documents with Claude Code integration via MCP.

## Installation

pipx - preferred!
```bash
pipx install okb
```

Or pip:
```bash
pip install okb
```

## Quick Start

```bash
# 1. Start the database
okb db start

# 2. (Optional) Deploy Modal embedder for faster batch ingestion
okb modal deploy

# 3. Ingest your documents
okb ingest ~/notes ~/docs

# 4. Configure Claude Code MCP (see below)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `okb db start` | Start pgvector database container |
| `okb db stop` | Stop database container |
| `okb db status` | Show database status |
| `okb db destroy` | Remove container and volume (destructive) |
| `okb ingest <paths>` | Ingest documents into knowledge base |
| `okb ingest <paths> --local` | Ingest using local GPU/CPU embedding (no Modal) |
| `okb serve` | Start MCP server (stdio, for Claude Code) |
| `okb serve --http` | Start HTTP MCP server with token auth |
| `okb watch <paths>` | Watch directories for changes |
| `okb config init` | Create default config file |
| `okb config show` | Show current configuration |
| `okb modal deploy` | Deploy GPU embedder to Modal |
| `okb token create` | Create API token for HTTP server |
| `okb token list` | List tokens for a database |
| `okb token revoke` | Revoke an API token |
| `okb sync list` | List available API sources (plugins) |
| `okb sync list-projects <source>` | List projects from source (for config) |
| `okb sync run <sources>` | Sync data from external APIs |
| `okb sync auth <source>` | Interactive OAuth setup (e.g., dropbox-paper) |
| `okb sync status` | Show last sync times |
| `okb rescan` | Check indexed files for changes, re-ingest stale |
| `okb rescan --dry-run` | Show what would change without executing |
| `okb rescan --delete` | Also remove documents for missing files |
| `okb llm status` | Show LLM config and connectivity |
| `okb llm deploy` | Deploy Modal LLM for open model inference |
| `okb llm clear-cache` | Clear LLM response cache |


## Configuration

Configuration is loaded from `~/.config/okb/config.yaml` (or `$XDG_CONFIG_HOME/okb/config.yaml`).

Create default config:
```bash
okb config init
```

Example config:
```yaml
databases:
  personal:
    url: postgresql://knowledge:localdev@localhost:5433/personal_kb
    default: true    # Used when --db not specified (only one can be default)
    managed: true    # okb manages via Docker
  work:
    url: postgresql://knowledge:localdev@localhost:5433/work_kb
    managed: true

docker:
  port: 5433
  container_name: okb-pgvector

chunking:
  chunk_size: 512
  chunk_overlap: 64
```

Use `--db <name>` to target a specific database with any command.

Environment variables override config file settings:
- `KB_DATABASE_URL` - Database connection string
- `OKB_DOCKER_PORT` - Docker port mapping
- `OKB_CONTAINER_NAME` - Docker container name

### Project-Local Config

Override global config per-project with `.okbconf.yaml` (searched from CWD upward):

```yaml
# .okbconf.yaml
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
| `modal` | `okb llm deploy` | ~$0.02/min GPU |

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
okb llm status              # Show config and connectivity
okb llm deploy              # Deploy Modal LLM (for provider: modal)
okb llm clear-cache         # Clear response cache
```

## Claude Code MCP Config

### stdio mode (default)

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "okb",
      "args": ["serve"]
    }
  }
}
```

### HTTP mode (for remote/shared servers)

First, start the HTTP server and create a token:

```bash
# Create a token
okb token create --db default -d "Claude Code"
# Output: okb_default_rw_a1b2c3d4e5f6g7h8

# Start HTTP server
okb serve --http --host 0.0.0.0 --port 8080
```

Then configure Claude Code to connect via SSE:

```json
{
  "mcpServers": {
    "knowledge-base": {
      "type": "sse",
      "url": "http://localhost:8080/sse",
      "headers": {
        "Authorization": "Bearer okb_default_rw_a1b2c3d4e5f6g7h8"
      }
    }
  }
}
```

## MCP Tools available to LLM

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
| `get_database_info` | Get database description, topics, and stats |
| `set_database_description` | Update database description/topics (LLM can self-document) |
| `add_todo` | Create a TODO item in the knowledge base |
| `trigger_sync` | Sync API sources (Todoist, GitHub, Dropbox Paper) |
| `trigger_rescan` | Check indexed files for changes and re-ingest |

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


## Plugin System

OKB supports plugins for custom file parsers and API data sources (GitHub, Todoist, etc).

### Creating a Plugin

```python
# File parser plugin
from okb.plugins import FileParser, Document

class EpubParser:
    extensions = ['.epub']
    source_type = 'epub'

    def can_parse(self, path): return path.suffix.lower() == '.epub'
    def parse(self, path, extra_metadata=None) -> Document: ...

# API source plugin
from okb.plugins import APISource, SyncState, Document

class GitHubSource:
    name = 'github'
    source_type = 'github-issue'

    def configure(self, config): ...
    def fetch(self, state: SyncState | None) -> tuple[list[Document], SyncState]: ...
```

### Registering Plugins

In your plugin's `pyproject.toml`:
```toml
[project.entry-points."okb.parsers"]
epub = "okb_epub:EpubParser"

[project.entry-points."okb.sources"]
github = "okb_github:GitHubSource"
```

### Configuring API Sources

```yaml
# ~/.config/okb/config.yaml
plugins:
  sources:
    github:
      enabled: true
      token: ${GITHUB_TOKEN}  # Resolved from environment
      repos: [owner/repo1, owner/repo2]
    todoist:
      enabled: true
      token: ${TODOIST_TOKEN}
      include_completed: false     # Sync completed tasks
      completed_days: 30           # Days of completed history
      include_comments: false      # Include task comments (1 API call per task)
      project_filter: []           # List of project IDs (use sync list-projects to find)
    dropbox-paper:
      enabled: true
      # Option 1: Refresh token (recommended, auto-refreshes)
      app_key: ${DROPBOX_APP_KEY}
      app_secret: ${DROPBOX_APP_SECRET}
      refresh_token: ${DROPBOX_REFRESH_TOKEN}
      # Option 2: Access token (short-lived, expires after ~4 hours)
      # token: ${DROPBOX_TOKEN}
      folders: [/]            # Optional: filter to specific folders
```

**Dropbox Paper OAuth Setup:**
```bash
okb sync auth dropbox-paper
```
This interactive command will guide you through getting a refresh token from Dropbox.

## License

MIT
