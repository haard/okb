# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
