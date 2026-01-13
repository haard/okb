# Personal Knowledge Base for Claude Code

A local-first semantic search system using Modal for GPU embedding, pgvector for storage, and MCP for Claude Code integration.

## Key Features

- **Local-first**: pgvector and MCP server run on your machine — zero cloud cost for daily use
- **GPU burst via Modal**: On-demand embedding generation (~$0.02 per 1000 chunks)
- **Contextual chunking**: Documents are chunked with title, project, section, and tag context for better retrieval
- **Claude Code integration**: Semantic search directly from your terminal via MCP
- **Hybrid search**: Combines semantic similarity with keyword matching

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

## Quick Start

```bash
# 1. Clone and enter directory
cd knowledge-base

# 2. Run setup (starts pgvector, installs dependencies)
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Deploy Modal embedder (one-time)
modal setup
modal deploy modal_embedder.py

# 4. Ingest your documents
source .venv/bin/activate
python ingest.py ~/notes ~/docs

# 5. Configure Claude Code (see setup output for config)

# 6. (Optional) Watch for changes
python scripts/watch.py ~/notes
```

## Contextual Chunking

The key insight: embedding "Use `select_related()` to avoid N+1 queries" alone loses context. The system automatically adds:

```
Document: Django Performance Notes
Project: student-app          ← inferred from path or frontmatter
Section: Query Optimization   ← extracted from markdown headers
Topics: django, performance   ← from frontmatter tags
Content: Use `select_related()` to avoid N+1 queries...
```

This contextual text is what gets embedded. The original chunk is stored separately for display.

### Context Sources

| Source | How It's Used |
|--------|---------------|
| **Document title** | Always included |
| **Project** | From frontmatter `project:` or inferred from path (`~/projects/{name}/...`) |
| **Section headers** | Extracted from markdown `## Heading` structure |
| **Tags** | From frontmatter `tags: [...]` |
| **Code structure** | Classes and functions extracted for code files |

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

## File Structure

```
knowledge-base/
├── docker-compose.yml      # pgvector container
├── init.sql                # Database schema
├── pyproject.toml          # Python project config
├── config.py               # Shared configuration
├── modal_embedder.py       # Modal GPU embedding service
├── local_embedder.py       # CPU embedding for queries
├── ingest.py               # Document ingestion pipeline
├── mcp_server.py           # MCP server for Claude Code
└── scripts/
    ├── setup.sh            # Initial setup
    └── watch.py            # File watcher for auto-updates
```

## Usage

### Ingesting Documents

```bash
# Ingest a directory
python ingest.py ~/notes

# Ingest with explicit project metadata
python ingest.py ~/projects/myapp --metadata '{"project": "myapp"}'

# Use local CPU embedding (slower, no Modal needed)
python ingest.py ~/notes --local
```

### MCP Tools (Available in Claude Code)

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Semantic search with natural language queries |
| `keyword_search` | Exact keyword/symbol matching |
| `hybrid_search` | Combined semantic + keyword (RRF fusion) |
| `get_document` | Retrieve full document by path |
| `list_sources` | Show indexed document stats |
| `list_projects` | List known projects |
| `recent_documents` | Show recently indexed files |

### Example Queries in Claude Code

```
> Search my notes for Django query optimization techniques

> Find code related to authentication middleware

> What do I have documented about PostgreSQL vacuum?

> Show me recent documents in the student-app project
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_DATABASE_URL` | `postgresql://knowledge:localdev@localhost:5433/knowledge_base` | Database connection |
| `PGVECTOR_PASSWORD` | `localdev` | PostgreSQL password |

### Claude Code MCP Config

Add to `~/.claude.json` (or your Claude Code config location):

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/path/to/knowledge-base/.venv/bin/python",
      "args": ["/path/to/knowledge-base/mcp_server.py"],
      "env": {
        "KB_DATABASE_URL": "postgresql://knowledge:localdev@localhost:5433/knowledge_base"
      }
    }
  }
}
```

## Cost Estimate

| Component | Local | Cloud Alternative |
|-----------|-------|-------------------|
| pgvector | $0 | ~$15-30/mo (CloudSQL) |
| MCP Server | $0 | ~$5/mo (small VM) |
| Modal embedding | ~$0.50-2/mo | N/A |
| **Total** | **~$1-2/mo** | **~$20-35/mo** |

Modal pricing: ~$0.000164/sec for T4 GPU

## Scaling Notes

| Scale | Chunks | RAM | Disk | Notes |
|-------|--------|-----|------|-------|
| Personal | <10k | 512 MB | 1 GB | Laptop-friendly |
| Medium | <100k | 1 GB | 5 GB | Typical knowledge base |
| Large | <1M | 2-4 GB | 20 GB | Extensive archives |

The HNSW index in pgvector handles 100k+ chunks efficiently. For larger scales, consider:
- Increasing `m` and `ef_construction` in index parameters
- Using a dedicated PostgreSQL instance
- Partitioning by project/source_type

## Development

```bash
# Run tests
pytest

# Lint
ruff check .

# Format
ruff format .
```

## Future Enhancements

- [ ] PDF support via PyMuPDF
- [ ] Web page clipper
- [ ] Incremental sync (track file mtimes)
- [ ] Multi-user with authentication
- [ ] Scheduled backups

## License

MIT
