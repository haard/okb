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
| `okb db migrate [name]` | Apply pending migrations (optionally for specific db) |
| `okb db list` | List configured databases |
| `okb db destroy` | Remove container and volume (destructive) |
| `okb db snapshot save [name]` | Create database snapshot (default: timestamp) |
| `okb db snapshot list` | List available snapshots |
| `okb db snapshot restore <name>` | Restore from snapshot (creates pre-restore backup) |
| `okb db snapshot restore <name> --no-backup` | Restore without pre-restore backup |
| `okb db snapshot delete <name>` | Delete a snapshot |
| `okb ingest <paths>` | Ingest documents into knowledge base |
| `okb ingest <paths> --local` | Ingest using local GPU/CPU embedding (no Modal) |
| `okb serve` | Start MCP server (stdio, for Claude Code) |
| `okb serve --http` | Start HTTP MCP server with token auth |
| `okb watch <paths>` | Watch directories for changes |
| `okb config init` | Create default config file |
| `okb config show` | Show current configuration |
| `okb config path` | Print config file path |
| `okb modal deploy` | Deploy GPU embedder to Modal |
| `okb token create` | Create API token for HTTP server |
| `okb token list` | List tokens for a database |
| `okb token revoke [TOKEN] --id <n>` | Revoke token by full value or ID |
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
| `okb enrich run` | Extract TODOs and entities from documents |
| `okb enrich run --dry-run` | Show what would be enriched |
| `okb enrich pending` | List entities awaiting review |
| `okb enrich approve <id>` | Approve a pending entity |
| `okb enrich reject <id>` | Reject a pending entity |
| `okb enrich analyze` | Analyze database and update description/topics |
| `okb enrich consolidate` | Run entity consolidation (duplicates, clusters) |
| `okb enrich merge-proposals` | List pending merge proposals |
| `okb enrich approve-merge <id>` | Approve an entity merge |
| `okb enrich reject-merge <id>` | Reject an entity merge |
| `okb enrich clusters` | List topic clusters |
| `okb enrich relationships` | List entity relationships |
| `okb schedule add <source> <interval>` | Schedule periodic sync via systemd timer |
| `okb schedule remove <source>` | Remove a scheduled sync timer |
| `okb schedule list` | List all active sync timers |
| `okb service install` | Install systemd user services for background operation |
| `okb service uninstall` | Remove systemd user services |
| `okb service status` | Show service status |
| `okb service start` | Start okb services |
| `okb service stop` | Stop okb services |
| `okb service restart` | Restart services (use after upgrading okb) |
| `okb service logs [-f]` | Show service logs (optionally follow) |


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
- `OKB_DATABASE_URL` - Database connection string
- `OKB_DOCKER_PORT` - Docker port mapping
- `OKB_CONTAINER_NAME` - Docker container name
- `OKB_SERVER_URL` - Remote server URL (overrides default server)
- `OKB_TOKEN` - Remote server token (overrides default server)

**Config file permissions**: Config files must be mode 0600 (not readable by group/other)
since they may contain secrets. OKB checks on load and errors if too open.

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

### Remote Servers (Client Mode)

Connect to remote OKB HTTP servers:

```yaml
servers:
  personal:
    url: http://localhost:8080/mcp
    token: ${OKB_PERSONAL_TOKEN}
    default: true
  work:
    url: http://work-host:8080/mcp
    token: ${OKB_WORK_TOKEN}
```

Only one server can be `default: true`. If none is marked, the first is used.

Local config can override the default server per-project:
```yaml
# .okbconf.yaml
default_server: work
```

### Per-Database Source Overrides

Databases can override global plugin source configs (full replacement per source, no merge):

```yaml
databases:
  work:
    url: postgresql://...
    managed: true
    sources:
      github:
        enabled: true
        token: ${WORK_GITHUB_TOKEN}
      todoist:
        enabled: false
```

### LLM Integration (Optional)

Enable LLM-based document classification, filtering, and enrichment:

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

**Modal LLM Setup** (no API key needed, runs on Modal's GPUs):

```yaml
llm:
  provider: modal
  model: microsoft/Phi-3-mini-4k-instruct  # Recommended: no gating
```

Non-gated models (work immediately):
- `microsoft/Phi-3-mini-4k-instruct` - Good quality, 4K context
- `Qwen/Qwen2-1.5B-Instruct` - Smaller/faster

Gated models (require HuggingFace approval + token):
- `meta-llama/Llama-3.2-3B-Instruct` - Requires accepting license at HuggingFace
- Setup: `modal secret create huggingface HF_TOKEN=hf_...`

Deploy after configuring:
```bash
okb llm deploy
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

### Document Enrichment

Extract TODOs and entities (people, projects, technologies) from documents using LLM:

```bash
okb enrich run                      # Enrich un-enriched documents
okb enrich run --dry-run            # Preview what would be enriched
okb enrich run --source-type markdown  # Only markdown files
okb enrich run --query "meeting"    # Filter by semantic search
```

Entities are created as pending suggestions for review:
```bash
okb enrich pending                  # List pending entities
okb enrich approve <id>             # Approve → creates entity document
okb enrich reject <id>              # Reject → hidden from future suggestions
```

Configure enrichment behavior:
```yaml
enrichment:
  enabled: true
  extract_todos: true
  extract_entities: true
  auto_create_todos: true       # TODOs created immediately
  auto_create_entities: false   # Entities go to pending review
  min_confidence_todo: 0.7
  min_confidence_entity: 0.8
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

The server uses Streamable HTTP transport (RFC 9728 compliant):
- `POST /mcp` - Send JSON-RPC messages, receive SSE response
- `GET /mcp` - Establish SSE connection for server notifications
- `DELETE /mcp` - Terminate session
- `/sse` is an alias for `/mcp` for backward compatibility

Configure your MCP client to connect:

```json
{
  "mcpServers": {
    "knowledge-base": {
      "type": "sse",
      "url": "http://localhost:8080/mcp",
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
| `save_knowledge` | Save knowledge from Claude (`source_type`: `claude-note` or `synthesis`) |
| `delete_knowledge` | Delete a Claude-saved knowledge entry |
| `get_actionable_items` | Query tasks/events with structured filters |
| `get_database_info` | Get database description, topics, and stats |
| `set_database_description` | Update database description/topics (LLM can self-document) |
| `add_todo` | Create a TODO item in the knowledge base |
| `trigger_sync` | Sync API sources (Todoist, GitHub, Dropbox Paper). Accepts `repos` for GitHub. |
| `trigger_rescan` | Check indexed files for changes and re-ingest |
| `list_sync_sources` | List available API sync sources with status |
| `enrich_document` | Run LLM enrichment to extract TODOs/entities |
| `list_pending_entities` | List entities awaiting review |
| `approve_entity` | Approve a pending entity |
| `reject_entity` | Reject a pending entity |
| `analyze_knowledge_base` | Analyze content and generate description/topics |
| `get_synthesis_samples` | Get document samples and stats for LLM-driven synthesis |
| `find_entity_duplicates` | Find potential duplicate entities |
| `merge_entities` | Merge duplicate entities |
| `list_pending_merges` | List pending merge proposals |
| `approve_merge` | Approve a merge proposal |
| `reject_merge` | Reject a merge proposal |
| `get_topic_clusters` | Get topic clusters from consolidation |
| `get_entity_relationships` | Get relationships between entities |
| `run_consolidation` | Run full entity consolidation pipeline |

## Claude.ai Integration (OAuth Shim)

Claude.ai requires OAuth 2.1 for MCP server connections. The `oauth/` directory contains a
Cloudflare Worker that bridges OAuth 2.1 to OKB's bearer token auth:

```
Claude.ai ──OAuth 2.1──▶ Cloudflare Worker ──Bearer token──▶ OKB HTTP server
                              │
                         GitHub login → maps to pre-existing OKB token
```

See [`oauth/README.md`](oauth/README.md) for setup instructions.

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

# Your Document Title

Content here...
```

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
