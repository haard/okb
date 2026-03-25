# Owned Knowledge Base (OKB)

A local-first semantic search system for personal documents with Claude Code integration via MCP.

## Architecture

```
Ingestion:  Files → Contextual Chunking → Embedding (Modal GPU or local CPU) → pgvector
Retrieval:  Claude Code → MCP Server → CPU Embedding → pgvector → Results
Claude.ai:  Claude.ai → OAuth 2.1 → CF Worker (oauth/) → Bearer token → OKB HTTP server
```

## Package Structure

```
okb/
├── __init__.py
├── cli.py           # Click-based CLI (entry point: okb)
├── config.py        # Configuration, DatabaseConfig, ServerConfig, lazy global config
├── http_server.py   # HTTP transport server with token authentication
├── ingest.py        # Document ingestion pipeline
├── local_embedder.py # CPU-based embedding for queries
├── mcp_server.py    # MCP server for Claude Code (stdio transport)
├── migrate.py       # Database migration runner (yoyo-migrations)
├── modal_embedder.py # GPU embedding service on Modal
├── rescan.py        # Check indexed files for changes, re-ingest stale
├── tokens.py        # API token generation, storage, and verification
├── tools.py         # Shared tool dispatch (execute_tool) and formatting helpers
├── llm/             # LLM integration for document classification and synthesis
│   ├── __init__.py  # Package exports, get_llm(), complete()
│   ├── base.py      # Protocol definitions, LLMResponse type
│   ├── providers.py # Claude provider implementation
│   ├── cache.py     # Response caching
│   ├── analyze.py   # Database/project-level analysis
│   └── synthesize.py # Knowledge synthesis (propose + approve documents)
├── plugins/         # Plugin system for custom parsers and API sources
│   ├── __init__.py  # Exports: FileParser, APISource, SyncState, Document, PluginRegistry
│   ├── base.py      # Protocol definitions
│   ├── registry.py  # Plugin discovery via entry_points
│   └── sources/     # Built-in API sources
│       ├── dropbox_paper.py  # Dropbox Paper sync
│       ├── github.py         # GitHub repo sync
│       └── todoist.py        # Todoist task sync
├── oauth/           # Cloudflare Worker: OAuth 2.1 shim for Claude.ai
│   ├── src/
│   │   ├── index.ts          # Worker entry — wires OAuthProvider + proxy handler
│   │   ├── github-handler.ts # /authorize → GitHub, /callback → token lookup
│   │   └── types.ts          # Env and UserProps types
│   ├── wrangler.toml         # Cloudflare Worker config
│   └── package.json
├── data/
│   └── init.sql     # PostgreSQL/pgvector schema (reference only)
├── migrations/
│   └── *.sql        # Versioned schema migrations
└── scripts/
    └── watch.py     # File watcher for auto-ingestion
```

## CLI Commands

```bash
# Database management
okb db start          # Start pgvector container (auto-runs migrations for all DBs)
okb db stop           # Stop container
okb db status         # Show status and migration info
okb db migrate        # Apply migrations to all databases
okb db migrate <name> # Apply migrations to specific database
okb db list           # List all configured databases
okb db destroy        # Remove container and volume

# Snapshots (backup/restore for managed databases)
okb db snapshot save [name]     # Create snapshot (default: timestamp)
okb db snapshot list            # List available snapshots
okb db snapshot restore <name>  # Restore from snapshot (creates pre-restore backup)
okb db snapshot restore <name> --no-backup  # Restore without pre-restore backup
okb db snapshot delete <name>   # Delete a snapshot

# Ingestion (use --db to target specific database)
okb ingest <paths>              # Ingest files/dirs (Modal GPU)
okb ingest <paths> --local      # Ingest with CPU embedding
okb ingest <url>                # Ingest web page (requires [web] extra)
okb ingest <paths> --db work    # Ingest to named database

# Client-side ingestion (via remote HTTP server)
okb ingest <paths>                        # Parse locally, ingest via server
okb ingest <paths> --project myproject    # Set project for all documents
okb ingest <paths> -m '{"key":"val"}'     # Attach extra metadata

# MCP server (use --db to serve specific database)
okb serve             # Serve default database (stdio transport)
okb serve --db work   # Serve named database
okb serve --http      # HTTP transport on localhost:8080
okb serve --http --host 0.0.0.0 --port 9000  # Custom host/port

# Token management (for HTTP server authentication)
okb token create --db default              # Create read-write token
okb token create --db personal --ro        # Create read-only token
okb token create --db work -d "Web UI"     # With description
okb token list --db default                # List tokens for database
okb token revoke <full_token>              # Revoke a token

# File watching
okb watch <paths>            # Watch with default database
okb watch <paths> --db work  # Watch with named database

# Configuration
okb config init       # Create ~/.config/okb/config.yaml
okb config show       # Show current config

okb modal deploy      # Deploy embedder to Modal

# Plugin sync (API sources)
okb sync list                   # List available API sources
okb sync list-projects <source> # List projects from source (for config)
okb sync status                 # Show last sync times
okb sync run <sources>          # Sync specific sources
okb sync run --all              # Sync all enabled sources
okb sync run dropbox-paper --full        # Full sync (ignore incremental state)
okb sync run dropbox-paper --db work     # Sync to specific database

# Todoist sync
okb sync run todoist                     # Sync tasks from Todoist
okb sync list-projects todoist           # List Todoist projects with IDs

# GitHub sync (requires --repo)
okb sync run github --repo owner/repo              # Sync README + docs/ (default)
okb sync run github --repo org/r1 --repo org/r2    # Multiple repos
okb sync run github --repo owner/repo --issues     # Include issues
okb sync run github --repo owner/repo --prs        # Include pull requests
okb sync run github --repo owner/repo --wiki       # Include wiki pages
okb sync run github --repo owner/repo --source     # All source files (not just README+docs)
okb sync run github --repo owner/repo --issues --prs --wiki  # Everything

# Rescan for changes
okb rescan                      # Check indexed files for freshness, re-ingest changed
okb rescan --dry-run            # Show what would change without executing
okb rescan --delete             # Also remove documents for missing files
okb rescan --db work --local    # Rescan specific database with local embedding

# Knowledge synthesis (generate topic summaries and insights)
okb synthesize run                       # Generate synthesis proposals
okb synthesize run --project myproj      # Scope to specific project
okb synthesize run --max-proposals 5     # Fewer proposals
okb synthesize run --dry-run             # Preview what would be sampled
okb synthesize pending                   # List pending proposals
okb synthesize approve <id>              # Approve a proposal
okb synthesize reject <id>               # Reject a proposal
okb synthesize review                    # Interactive review loop (A/E/R/S/Q)

# Database analysis (extract themes)
okb synthesize analyze                   # Analyze and update description/topics
okb synthesize analyze --stats-only      # Show stats without LLM call
okb synthesize analyze --project myproj  # Analyze specific project
okb synthesize analyze --no-update       # Analyze without saving to metadata

# Scheduled sync (systemd user timers)
okb schedule add todoist 1h              # Sync todoist every hour (default db)
okb schedule add github 6h --db work    # Sync github every 6h for work db
okb schedule remove todoist              # Remove todoist sync schedule
okb schedule remove github --db work    # Remove schedule for specific db
okb schedule list                        # List all active sync timers

# Systemd user services (background operation)
okb service install           # Install and start systemd user services
okb service install --no-start  # Install without starting
okb service uninstall         # Stop and remove services
okb service status            # Show service status
okb service start             # Start services
okb service stop              # Stop services
okb service restart           # Restart services (use after upgrading okb)
okb service logs              # Show service logs
okb service logs -f           # Follow logs
# Environment file for services: ~/.config/okb/env (created by service install)
# Set API keys there, e.g. ANTHROPIC_API_KEY=sk-ant-...
```

## Configuration

Config file: `~/.config/okb/config.yaml` (or `$XDG_CONFIG_HOME/okb/config.yaml`)

Priority: Environment variables > config file > defaults

### Multiple Databases

Support for multiple knowledge bases (personal, work, shared):

```yaml
databases:
  personal:
    url: postgresql://knowledge:localdev@localhost:5433/personal_kb
    default: true    # Used when --db not specified
    managed: true    # okb manages via Docker container
    description: "Personal notes and documents"  # Helps LLM route queries
    topics: [notes, journal, recipes]            # Keywords for LLM context
  work:
    url: postgresql://knowledge:localdev@localhost:5433/work_kb
    managed: true
    description: "Work projects, meeting notes, technical documentation"
    topics: [work, projects, meetings, tech]
  shared:
    url: postgresql://shared-server:5432/team_kb
    managed: false   # External server, not managed by okb
```

The `description` and `topics` fields help LLMs understand what kind of content is in each database,
enabling better query routing when multiple knowledge bases are available. These are sent as MCP
server instructions for stdio transport, and available via `get_database_info` tool for HTTP.

**Backward compatibility**: Legacy single `database_url` config still works.

**Note**: Only one database can have `default: true`. If multiple are marked, config loading fails.

### Project-Local Config Overlay

Projects can override global config with a `.okbconf.yaml` file. OKB searches from CWD upward.

```yaml
# .okbconf.yaml
default_database: work  # Use 'work' db for this project

extensions:
  skip_directories:     # Extends global list
    - test_fixtures
    - migrations

security:
  skip_patterns:        # Extends global list
    - "*.generated.ts"

plugins:
  sources:
    dropbox-paper:
      enabled: false    # Override for this project
```

Merge strategy:
- **Scalars**: Local replaces global
- **Lists**: Local extends global
- **Dicts**: Deep merge (local takes precedence)

Type mismatches (e.g., list in local vs dict in global) raise an error.

### MCP Multi-Database Setup

Run separate MCP server instances per database:

```json
{
  "mcpServers": {
    "okb-personal": {
      "command": "okb",
      "args": ["serve", "--db", "personal"]
    },
    "okb-work": {
      "command": "okb",
      "args": ["serve", "--db", "work"]
    }
  }
}
```

### Remote Servers (Client Mode)

Connect to remote OKB HTTP servers as a client. Used by the `okb` CLI client commands.

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

- Only one server can have `default: true`; if none marked, first is used
- `Config.get_server(name)` returns `ServerConfig` (name, url, token, default)

Environment variables: `OKB_SERVER_URL`, `OKB_TOKEN` (override the default server)

Local config overlay supports `default_server`:
```yaml
# .okbconf.yaml
default_server: work  # Use 'work' server in this project
```

### Per-Database Source Overrides

Databases can override global plugin source configs. Per-database config fully replaces the
global config for that source (no deep merge):

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
        enabled: false  # Disable todoist for work db
```

`Config.get_source_config(source_name, db_name)` and `Config.list_enabled_sources(db_name)`
resolve per-database overrides automatically.

### HTTP Server Settings

```yaml
http:
  host: 127.0.0.1   # Default: localhost only
  port: 8080        # Default port
```

Environment variables: `OKB_HTTP_HOST`, `OKB_HTTP_PORT`

### Other Settings

- `docker.port` / `OKB_DOCKER_PORT`
- `docker.container_name` / `OKB_CONTAINER_NAME`

### Config File Permissions

Config files (`config.yaml`, `.okbconf.yaml`) must not be readable by group/other (mode 0600)
since they may contain secrets. OKB checks permissions on load and raises
`InsecureConfigError` if too open. Fix with `chmod 600 <path>`.

### LLM Settings

Optional LLM integration for document classification and synthesis.

```yaml
llm:
  provider: claude          # "claude", "modal", or null (disabled)
  model: claude-haiku-4-5-20251001
  timeout: 30               # Request timeout in seconds
  cache_responses: true     # Cache responses by content hash
  # For AWS Bedrock instead of direct API:
  use_bedrock: false
  aws_region: us-west-2
```

Environment variables:
- `OKB_LLM_PROVIDER` - Provider name
- `OKB_LLM_MODEL` - Model name
- `OKB_LLM_TIMEOUT` - Timeout in seconds
- `ANTHROPIC_API_KEY` - API key for direct Claude API access

Install dependencies: `pip install 'okb[llm]'` (or `[llm-bedrock]` for Bedrock)

CLI commands:
```bash
okb llm status              # Show config and connectivity
okb llm clear-cache         # Clear response cache
okb llm clear-cache --older-than 30  # Clear entries older than 30 days
okb llm deploy              # Deploy Modal LLM app (for provider: modal)
```

#### LLM Providers

| Provider | Config | Cost | Notes |
|----------|--------|------|-------|
| `claude` | `ANTHROPIC_API_KEY` | ~$0.25/1M tokens | Fast, high quality |
| `modal` | `okb llm deploy` | ~$0.02/min GPU | Open models, no API key |

**Claude provider** (recommended for synthesis):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Modal provider** (open models on Modal GPUs):
```yaml
llm:
  provider: modal
  model: microsoft/Phi-3-mini-4k-instruct  # Recommended: no gating required
```

Then deploy: `okb llm deploy`

**Modal model options:**

| Model | Gated | Notes |
|-------|-------|-------|
| `microsoft/Phi-3-mini-4k-instruct` | No | Good quality, 4K context |
| `Qwen/Qwen2-1.5B-Instruct` | No | Smaller/faster |
| `meta-llama/Llama-3.2-3B-Instruct` | Yes | Requires HF approval |

For gated models (Llama, etc.):
1. Accept license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Create HF token at https://huggingface.co/settings/tokens
3. Add to Modal: `modal secret create huggingface HF_TOKEN=hf_...`

#### Pre-ingest Filtering

Filter documents during sync based on LLM classification:

```yaml
plugins:
  sources:
    dropbox-paper:
      enabled: true
      llm_filter:
        enabled: true
        prompt: "Skip meeting notes and drafts"  # Optional custom instructions
        action_on_skip: discard  # "discard" or "archive"
```

The LLM classifies each document as:
- `ingest` - Process normally
- `skip` - Don't ingest (based on `action_on_skip`)
- `review` - Ingest but flag for review

### Knowledge Synthesis

LLM-based database-level synthesis generates useful reference documents by analyzing
the knowledge base broadly and proposing topic summaries, entity profiles, relationship
maps, and cross-cutting insights.

**Requires LLM provider** - configure `llm.provider` or set `ANTHROPIC_API_KEY`.

CLI commands:
```bash
okb synthesize run                        # Generate proposals from database
okb synthesize run --project myproject    # Scope to project
okb synthesize run --max-proposals 5      # Limit proposals
okb synthesize pending                    # List pending proposals
okb synthesize approve <id>               # Approve -> creates searchable document
okb synthesize reject <id>                # Reject proposal
okb synthesize review                     # Interactive review (A/E/R/S/Q)
```

Approved synthesis documents:
- `source_path`: `okb://synthesis/{uuid}`
- `source_type`: `synthesis`
- Stored with embedding, searchable like any other document

## HTTP Server and Authentication

The MCP server can run in two modes:

1. **stdio** (default): For direct Claude Code integration
2. **HTTP**: For web clients, IDE plugins, or shared servers

### Token Format

```
okb_<database>_<ro|rw>_<random16hex>
Example: okb_personal_ro_a1b2c3d4e5f6g7h8
```

- Tokens are scoped to a specific database
- `ro` = read-only (search, list, get operations)
- `rw` = read-write (includes save_knowledge, update_knowledge, delete_knowledge)
- Tokens are stored hashed (SHA256) in the database `tokens` table

### HTTP Endpoints (Streamable HTTP Transport)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth required) |
| `/mcp` | GET, POST, DELETE | MCP endpoint (requires token) |
| `/sse` | GET, POST, DELETE | Alias for `/mcp` (requires token) |

The server uses Streamable HTTP transport (RFC 9728 compliant):
- `POST /mcp` - Send JSON-RPC messages, receive SSE response
- `GET /mcp` - Establish SSE connection for server notifications
- `DELETE /mcp` - Terminate session
- Session ID returned in `Mcp-Session-Id` response header

### Authentication

Tokens can be provided via:
1. **Authorization header** (preferred): `Authorization: Bearer <token>`
2. **Query parameter**: `?token=<token>`

### Client Usage

```bash
# Create a token
okb token create --db personal --ro -d "Web client"
# Output: okb_personal_ro_a1b2c3d4e5f6g7h8

# Start HTTP server
okb serve --http

# Health check
curl http://localhost:8080/health

# MCP clients connect to /mcp with token in Authorization header
# The MCP client handles the Streamable HTTP protocol automatically
```

### Permission Mapping

| Permission | Allowed Tools |
|------------|---------------|
| `ro` (read) | search_knowledge, keyword_search, hybrid_search, get_document, list_sources, list_projects, list_documents_by_project, recent_documents, get_actionable_items, get_database_info, get_synthesis_samples, list_pending_synthesis, list_snapshots |
| `rw` (write) | All read tools + save_knowledge, update_knowledge, delete_knowledge, set_database_description, add_todo, trigger_sync, trigger_rescan, ingest_documents, synthesize_knowledge, approve_synthesis, reject_synthesis, edit_pending_synthesis, analyze_knowledge_base, save_snapshot, restore_snapshot |

## MCP Tools Available

- `search_knowledge` - Semantic search with natural language (supports `since` filter)
- `keyword_search` - Full-text keyword search (supports `since` filter)
- `hybrid_search` - Combined semantic + keyword with RRF fusion (supports `since` filter)
- `get_document` - Retrieve full document by path
- `list_sources` - Show indexed document stats
- `list_projects` - List known projects
- `list_documents_by_project` - List all documents for a specific project
- `get_project_stats` - List projects with document counts (for consolidation)
- `rename_project` - Rename a project across all documents. Requires write permission.
- `set_document_project` - Set/clear project for a single document. Requires write permission.
- `recent_documents` - Show recently indexed files (shows document dates)
- `save_knowledge` - Save knowledge from Claude. Optional `source_type`: `claude-note` (default) or
  `synthesis` (uses `okb://synthesis/` path, excluded from future sampling)
- `update_knowledge` - Update an existing document in-place (title, content, tags, project). Requires write.
- `delete_knowledge` - Delete any document by source path. Requires write permission.
- `get_actionable_items` - Query tasks, events, emails with structured filters (daily briefs)
- `get_database_info` - Get database description, topics, and content statistics
- `set_database_description` - LLM can update description/topics based on content analysis
- `add_todo` - Create a TODO item in the knowledge base
- `trigger_sync` - Sync API sources (Todoist, GitHub, etc.). Accepts `repos` param for GitHub. Requires write.
- `trigger_rescan` - Check indexed files for changes and re-ingest. Requires write permission.
- `ingest_documents` - Ingest pre-parsed documents (chunking, embedding, storage). Requires write.
- `list_sync_sources` - List available API sync sources with status and last sync time
- `get_synthesis_samples` - Get document samples and stats for LLM-driven synthesis (read-only)
- `synthesize_knowledge` - Analyze DB and propose synthetic knowledge documents. Requires write.
- `list_pending_synthesis` - List pending synthesis proposals awaiting review
- `approve_synthesis` - Approve a proposal, creating a searchable document. Requires write.
- `reject_synthesis` - Reject a pending synthesis proposal. Requires write.
- `edit_pending_synthesis` - Edit a proposal before approve/reject. Requires write.
- `analyze_knowledge_base` - Analyze content to generate/update description and topics. Requires write.
- `save_snapshot` - Create database snapshot for backup. Requires write permission.
- `list_snapshots` - List available database snapshots with metadata.
- `restore_snapshot` - Restore from snapshot (requires confirm=true). Requires write permission.

The `since` parameter accepts relative ('7d', '30d', '6mo', '1y'), natural language
('last week', '3 months ago'), or ISO dates.

### Database Info Tools

`get_database_info` returns:
- Config-defined description and topics
- LLM-enhanced description and topics (if set)
- Content statistics (source types, document counts, token counts)
- List of projects

`set_database_description` allows the LLM to store enhanced metadata:
- `description` - A concise description of what the knowledge base contains
- `topics` - List of topic keywords characterizing the content

This enables LLMs to:
1. Call `get_database_info` first to understand what's available
2. Analyze content and call `set_database_description` to improve future query routing

### Analyze Knowledge Base Tool

`analyze_knowledge_base` performs automated analysis of the knowledge base:
- Samples documents using diverse embedding-based selection
- Uses LLM to synthesize a description and topic keywords
- Optionally updates `database_metadata` with results

Parameters:
- `project` - Analyze only a specific project (optional)
- `sample_size` - Number of documents to sample (default: 15)
- `auto_update` - Whether to update database metadata (default: true)

This automates what would otherwise be manual exploration + `set_database_description`.

### Actionable Items Tool

`get_actionable_items` supports these filters for daily brief queries:
- `item_type` - Filter by source type (e.g., 'todoist-task', 'gcal-event', 'gmail')
- `status` - Filter by status ('pending', 'completed', 'cancelled')
- `due_date` - Filter tasks by due date ('today', 'tomorrow', 'yesterday', 'this week',
  'next week', 'last week', 'this month', 'next month', natural language, or 'YYYY-MM-DD')
- `event_date` - Filter events by date (same formats as due_date)
- `min_priority` - Filter by priority (1=highest, returns items with priority <= value)

Results are ordered by due_date/event_start (soonest first), then by priority.

### Project Management Tools

LLMs can consolidate and manage projects via MCP:

1. **`get_project_stats`** - View projects with document counts to identify duplicates or similar names
2. **`rename_project`** - Bulk rename (e.g., consolidate "my-app" and "MyApp" into "my-app")
3. **`set_document_project`** - Fix individual documents with wrong project assignment

Example workflow for an LLM:
```
1. Call get_project_stats to see all projects
2. Identify similar names: "okb", "OKB", "owned-knowledge-base"
3. Call rename_project("OKB", "okb") to consolidate
4. Call rename_project("owned-knowledge-base", "okb") to consolidate
```

## Technical Details

### Embedding Model
- Model: `nomic-ai/nomic-embed-text-v1.5`
- Dimension: 768
- Requires task prefixes: `search_query:` for queries, `search_document:` for documents
- Requires `trust_remote_code=True`

### pgvector Index
- HNSW index with cosine distance (`vector_cosine_ops`)
- Parameters: `m=16, ef_construction=64` (defaults, good for <100k chunks)

### Chunking Strategy
- Default: 512 tokens (~2048 chars) with 64 token overlap
- Contextual embedding text includes: document title, project, section headers, tags

### Org-mode TODO Extraction
Org files produce multiple documents:
- **Primary document** (`source_type='org'`): Full file for semantic search
- **TODO documents** (`source_type='org-todo'`): Each TODO item as separate document

TODO documents include:
- `source_path`: `file.org::*TODO Heading text` (org-mode link format)
- `status`: 'pending' (TODO/WAITING/NEXT/SOMEDAY) or 'completed' (DONE/CANCELLED)
- `priority`: 1-3 from `[#A]`/`[#B]`/`[#C]` (1=highest), or 5 for SOMEDAY items
- `due_date`: From `DEADLINE:` or `SCHEDULED:` timestamps

This enables both:
- Semantic search: "what's the plan for PDF support?"
- Structured queries: `get_actionable_items(status='pending', due_date='this_week')`

### Document Date Tracking
- Extracts `document_date` from frontmatter fields: `date`, `created`, `modified`, `updated`, `pubdate`
- Falls back to file mtime (`file_modified_at`) if no frontmatter date
- Dates displayed in search results and `recent_documents` as relative time ("3d ago")
- `since` filter accepts relative ('7d', '30d', '6mo', '1y'), natural language ('last week',
  '3 months ago'), or ISO date

### Security: Blocked Files
Automatically skips sensitive and low-value files during ingestion:
- Private keys (`id_rsa`, `*.pem`, `*.key`)
- Credentials (`.env`, `*credentials*`, `.netrc`, `.pgpass`)
- Lockfiles (`package-lock.json`, `yarn.lock`, `uv.lock`, `poetry.lock`)
- Minified assets (`*.min.js`, `*.min.css`, `*.bundle.js`)
- Content scanning for embedded secrets (AWS keys, private key headers)

## Plugin System

OKB supports plugins for custom file parsers and API-based data sources.

### Plugin Types

**FileParser** - Add support for new file formats:
```python
from okb.plugins import FileParser, Document

class EpubParser:
    extensions = ['.epub']
    source_type = 'epub'

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == '.epub'

    def parse(self, path: Path, extra_metadata: dict | None = None) -> Document:
        # Parse and return Document
        ...
```

**APISource** - Sync data from external services:
```python
from okb.plugins import APISource, SyncState, Document
from datetime import datetime, timezone

class TodoistSource:
    name = 'todoist'
    source_type = 'todoist-task'

    def configure(self, config: dict) -> None:
        self._token = config['token']

    def fetch(self, state: SyncState | None = None) -> tuple[list[Document], SyncState]:
        # Fetch tasks and populate structured fields
        doc = Document(
            source_path=f"todoist://task/{task_id}",
            source_type='todoist-task',
            title=task.content,
            content=task.description or task.content,
            due_date=parse_due_date(task.due),  # datetime for "tasks due today" queries
            status='pending' if not task.is_completed else 'completed',
            priority=5 - task.priority,  # Todoist uses 1-4 (4=highest), normalize to 1-5
        )
        ...
```

### Structured Fields for Actionable Items

The `Document` dataclass supports structured fields for tasks, events, and emails:

| Field | Type | Description |
|-------|------|-------------|
| `due_date` | `datetime \| None` | Task deadline (enables "tasks due today" queries) |
| `event_start` | `datetime \| None` | Calendar event start time |
| `event_end` | `datetime \| None` | Calendar event end time |
| `status` | `str \| None` | Item status: 'pending', 'completed', 'cancelled' |
| `priority` | `int \| None` | Priority 1-5 (1=highest) |

API source plugins should populate these fields to enable structured queries via `get_actionable_items`.

### Plugin Registration

Plugins are discovered via Python entry_points. In your plugin's `pyproject.toml`:

```toml
[project.entry-points."okb.parsers"]
epub = "okb_epub:EpubParser"

[project.entry-points."okb.sources"]
todoist = "okb_todoist:TodoistSource"
```

### API Source Configuration

Configure API sources in `~/.config/okb/config.yaml`:

```yaml
plugins:
  sources:
    github:
      enabled: true
      token: ${GITHUB_TOKEN}      # Resolved from environment
    todoist:
      enabled: true
      token: ${TODOIST_TOKEN}
      include_completed: false    # Sync completed tasks (default: false)
      completed_days: 30          # Days of completed history (default: 30)
      include_comments: false     # Include task comments (expensive, default: false)
      project_filter: []          # List of project IDs to sync (empty = all)
    dropbox-paper:
      enabled: true
      # Option 1: Refresh token (recommended, auto-refreshes)
      app_key: ${DROPBOX_APP_KEY}
      app_secret: ${DROPBOX_APP_SECRET}
      refresh_token: ${DROPBOX_REFRESH_TOKEN}
      # Option 2: Access token (short-lived, will expire after ~4 hours)
      # token: ${DROPBOX_TOKEN}
      folders: [/]                # Optional: filter to specific folder paths
```

### Dropbox Paper OAuth Setup

To get refresh tokens for Dropbox Paper (recommended for long-lived access):

```bash
okb sync auth dropbox-paper
```

This interactive command will:
1. Prompt for your app key and secret (from https://www.dropbox.com/developers/apps)
2. Generate an authorization URL for you to visit
3. Exchange the authorization code for a refresh token
4. Output the environment variables and config snippet

Manual setup if needed:
1. Create an app at https://www.dropbox.com/developers/apps
2. Set permissions: `files.content.read`, `sharing.read`
3. Use the OAuth flow with `token_access_type=offline`

The refresh token doesn't expire and will automatically obtain new access tokens as needed.

**Note**: Todoist project IDs differ from URLs. Use `okb sync list-projects todoist` to find IDs.

Environment variable syntax:
- `${VAR}` - Required, errors if not set
- `${VAR:-default}` - Optional with fallback value

### Sync State

API sources support incremental syncing via `SyncState`:
- `last_sync` - Timestamp of last successful sync
- `cursor` - Pagination cursor for APIs that support it
- `extra` - Dict for source-specific state

State is stored in the database (`sync_state` table) and persists across restarts.

## Dependencies

- `psycopg[binary]` - PostgreSQL driver
- `pgvector` - pgvector Python bindings
- `sentence-transformers` - Embedding generation
- `mcp` - Model Context Protocol SDK
- `click` - CLI framework
- `pyyaml` - Config file parsing
- `watchdog` - File system watcher for auto-ingestion
- `modal` - GPU embedding service deployment
- `yoyo-migrations` - Database schema migrations

### Optional extras
- `[pdf]` - `pymupdf` for PDF support
- `[docx]` - `python-docx` for DOCX support
- `[web]` - `trafilatura` for URL ingestion
- `[todoist]` - `todoist-api-python` for Todoist sync

## Known Limitations

- **Keyword search**: PostgreSQL FTS splits underscored terms (e.g., `select_related` becomes
  `select` + `relat`). Use semantic search for code symbols, or ILIKE for exact matches.
- **First query latency**: ~10-30s to load embedding model on first query (cached after).

## Development

Package management uses Poetry.

```bash
# Install dependencies
poetry install

# Install with dev dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Lint and format
poetry run ruff check .
poetry run ruff format .

# Or with ruff auto-fix
poetry run ruff check --fix .
```

### Ruff Configuration
- Line length: 100
- Target: Python 3.11+
- Enabled rules: E (errors), F (pyflakes), I (isort), N (naming), W (warnings), UP (pyupgrade)
