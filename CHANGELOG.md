# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.3.3] - 2026-04-30
### Fixed
- Org-mode parsing: content before first heading (e.g., metadata lines) now uses filename
  as section header instead of `None`, fixing schema validation error when ingesting
  org files without initial `* Heading` line

## [2.3.2] - 2026-04-29
### Fixed
- HTTP server: `POST`/`DELETE` to `/legacy/sse` now routes to Streamable HTTP instead of
  spawning a ghost legacy SSE session, fixing hangs with clients (e.g. Mistral Le Chat)
  that probe both transports on the same URL
- HTTP server: `/legacy/messages` returns 405 on non-`POST` requests instead of falling
  through to the SSE message handler

## [2.3.1] - 2026-04-13
### Added
- Legacy SSE transport at `/legacy/sse` and `/legacy/messages` for older MCP clients that
  don't support Streamable HTTP
- OAuth 2.1 shim (Cloudflare Worker) for Claude.ai remote MCP connections via GitHub login
- Multi-database OAuth: `TOKEN_MAP` accepts `{"user": {"db1": "tok1", "db2": "tok2"}}` —
  shows a database picker page during OAuth when multiple databases are configured
- `wrangler.template.toml` for OAuth worker config (real `wrangler.toml` now gitignored)
- `okb schedule add <source> <interval>` — schedule periodic sync via systemd user timers
- `okb schedule remove <source>` — remove a scheduled sync timer
- `okb schedule list` — list all active sync timers
- Async sync runner (`sync_runner.py`) with `sync_status` migration for tracking sync state
- `list_sync_sources` now shows running/error status and last error message

### Security
- Read-only token permission check switched from deny-list to allow-list — unknown tools
  are now rejected for `ro` tokens instead of only blocking known write tools
- Snapshot name validation prevents path traversal in `save_snapshot` and `restore_snapshot`
- GitHub GIT_ASKPASS: set owner-only permissions before writing, quote token with `shlex`
- Tool error responses no longer leak exception details (logged internally, generic message
  returned to client)
- HTTP server session tokens: validate token matches on existing sessions, evict expired
  sessions on access
- LLM synthesis/analysis: sanitize project names to prevent prompt injection
- GitHub sync: validate repo format, reject path traversal in file paths
- Token generation uses 128-bit random hex (was 64-bit)

### Changed
- `trigger_sync` and `trigger_rescan` MCP tools now run asynchronously and return immediately
- HTTP server session management: TTL-based eviction (24h), capped at 1000 sessions


## [2.3.0] - 2026-03-11

### Added
- Natural language date parsing via `dateparser` for `since`, `due_date`, and `event_date`
  parameters (e.g. "last week", "3 months ago", "yesterday")
- New date range keywords: `yesterday`, `last week`, `this month`, `next month`
- `okb sync run slack` client CLI subcommand with `--channel` option for filtering by channel ID
- Client CLI `sync run` restructured as group with per-source subcommands (`github`, `todoist`,
  `dropbox-paper`, `slack`) — source-specific options grouped under their subcommand
- GitHub sync options (`--issues`, `--prs`, `--wiki`, `--source`) now available in client CLI
- MCP `trigger_sync` tool accepts `include_issues`, `include_prs`, `include_wiki`,
  `include_source`, `folders`, and `channels` parameters

### Changed
- Clarified `save_knowledge` vs `ingest_documents` tool descriptions to prevent LLM confusion —
  `save_knowledge` is for LLM-created content, `ingest_documents` is for external sources
- `ingest_documents` rejects `claude://` and `okb://` source paths with guidance to use
  `save_knowledge` instead

### Fixed
- Fix race condition in HTTP transport auth lookup — use per-request `request_context` instead of
  shared `_current_token_info` attribute
- Add per-KB asyncio locks for connection safety (psycopg connections are not thread-safe)
- Serialize all KB method calls under the lock in HTTP transport via `_run_with_lock`

## [2.3.0a2] - 2026-03-01

### Added
- Remote ingestion via `ingest_documents` MCP tool — clients can parse files locally and send
  to the server for chunking, embedding, and storage
- `okb ingest` client CLI command for ingesting local files into a remote server
- `Document.to_dict()` / `Document.from_dict()` serialization for remote transport
- `--branch` option for GitHub sync to sync a specific branch instead of the default

### Changed
- Remote ingest uses Modal GPU embedding when available, falls back to local CPU

## [2.2.3] - 2026-02-27

### Fixed
- Run blocking tool operations (`trigger_sync`, `trigger_rescan`, `synthesize_knowledge`,
  `analyze_knowledge_base`) in threads to avoid blocking the event loop and breaking the
  SSE transport
- Unwrap `ExceptionGroup` in CLI client to show actual error messages instead of
  "unhandled errors in a TaskGroup (1 sub-exception)"

## [2.2.2] - 2026-02-27

### Added
- `EnvironmentFile` support for systemd services (`~/.config/okb/env`) for API keys like
  `ANTHROPIC_API_KEY`; template created on `service install`, permissions enforced at 0600

## [2.2.1] - 2026-02-27

### Fixed
- Systemd service files used `okb` (client CLI) instead of `okb-admin` for `db start` and
  `serve --http`, causing service startup failures

## [2.2.0] - 2026-02-27

### Added
- Named remote server configs (`servers:` block in config) with `ServerConfig` dataclass
- `Config.get_server(name)` to resolve server by name or default
- `default_server` support in local config overlay (`.okbconf.yaml`)
- Environment variable overrides `OKB_SERVER_URL` and `OKB_TOKEN` for the default server
- Per-database source config overrides (`databases.*.sources`), full replacement per source
- `db_name` parameter on `Config.get_source_config()` and `Config.list_enabled_sources()`
- Config file permission checking — files must be mode 0600 (`InsecureConfigError` on violation)
- `okb config init` now generates a fully commented template instead of raw defaults

### Changed
- Global `config` instance is now lazily initialized via `_LazyConfig` proxy
- `get_default_config_yaml()` returns commented YAML template

### Fixed
- `local_default_server` was referenced outside the `if local_path:` scope (safe at runtime
  due to short-circuit, but fragile); moved extraction into the block alongside `local_default_db`

## [2.0.0] - 2026-02-13

### Added
- Knowledge synthesis system (`okb synthesize run/pending/approve/reject/review`)
- `get_synthesis_samples` MCP tool — returns document samples and stats for LLM-driven synthesis
- `synthesize_knowledge` MCP tool — server-side LLM synthesis with pending/approve workflow
- `list_pending_synthesis`, `approve_synthesis`, `reject_synthesis`, `edit_pending_synthesis`
  MCP tools for managing synthesis proposals
- `analyze_knowledge_base` CLI command (`okb synthesize analyze`) with `--stats-only` option
- `source_type` parameter on `save_knowledge` — `'synthesis'` uses `okb://synthesis/` paths,
  excluded from future sampling
- `excerpt_length` parameter on `get_document_samples()` (default 500, configurable up to 3000)
- `timeout` parameter on LLM provider `complete()` for per-request timeout overrides
- Migration 0011: `pending_synthesis` table

### Removed
- **Breaking:** Entity enrichment system (`okb enrich` commands, `enrich_document` MCP tool)
- **Breaking:** Entity consolidation system (`run_consolidation`, `find_entity_duplicates`,
  `merge_entities`, `approve_merge`, `reject_merge`, `get_topic_clusters`,
  `get_entity_relationships` MCP tools)
- **Breaking:** `list_pending_entities`, `approve_entity`, `reject_entity` MCP tools
- Enrichment config section (`enrichment.*` settings)
- `okb/llm/enrich.py`, `okb/llm/consolidate.py`, `okb/llm/extractors/` package
- Migration 0011 drops: `pending_entities`, `entity_refs`, `entity_aliases`,
  `pending_entity_merges`, `entity_relationships`, `topic_clusters`,
  `topic_cluster_members`, `pending_cluster_merges`, `cross_doc_entity_candidates`,
  `consolidation_runs` tables; `enriched_at` and `enrichment_version` columns from documents

### Changed
- `okb enrich` CLI group replaced by `okb synthesize`
- HTTP server `READ_ONLY_TOOLS` updated (removed entity/consolidation tools,
  added `get_synthesis_samples`)

## [1.3.0] - 2025-02-07

### Added
- Systemd user services for background operation (`okb service install/uninstall/status/start/stop/restart/logs`)
- `repos` parameter to `trigger_sync` MCP tool for GitHub sync without pre-configuration

## [1.2.0] - 2025-02-05

### Added
- Database snapshot commands (`okb db snapshot save/list/restore/delete`) for backup and restore
- MCP tools for snapshots: `save_snapshot`, `list_snapshots`, `restore_snapshot`
- Pre-restore backup: `snapshot restore` now automatically creates a backup before restoring
  - MCP: Always creates backup (safety first for LLM agents)
  - CLI: Creates backup by default, use `--no-backup` to skip
- Claude-generated unit tests, mainly for regression testing

## [1.1.2] - 2025-01-31

### Fixed
- Add missing project management tools to HTTP server (`get_project_stats`, `rename_project`,
  `set_document_project`)

## [1.1.1] - 2025-01-31

### Changed
- Switch HTTP server to Streamable HTTP transport (RFC 9728 compliant)

## [1.1.0] - 2025-01-30

### Added
- Entity consolidation tools: `find_entity_duplicates`, `merge_entities`, `list_pending_merges`,
  `approve_merge`, `reject_merge`
- Project management MCP tools: `get_project_stats`, `rename_project`, `set_document_project`,
  `list_documents_by_project`
- `enrich all` command runs full enrichment + consolidation pipeline
- Path pattern filtering for targeted enrichment
- Local CPU embedding option for entity approval (`--local` flag)
- ID-based token revocation

### Changed
- `delete_knowledge` now works on any document (not just `claude://` paths)
- Entity extraction improved to skip well-known technologies
