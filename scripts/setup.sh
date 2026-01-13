#!/usr/bin/env bash
# Setup script for the knowledge base
set -e

echo "=== Knowledge Base Setup ==="

# Check for required tools
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is required but not installed."
        exit 1
    fi
}

check_command docker
check_command python3

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$python_version < 3.11" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.11+ required, found $python_version"
    exit 1
fi

echo "✓ Prerequisites OK"

# Start pgvector
echo ""
echo "Starting pgvector..."
docker compose up -d

# Wait for database to be ready
echo "Waiting for database..."
for i in {1..30}; do
    if docker compose exec -T pgvector pg_isready -U knowledge -d knowledge_base &> /dev/null; then
        echo "✓ Database ready"
        break
    fi
    sleep 1
done

# Create virtual environment if needed
if [[ ! -d ".venv" ]]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
echo ""
echo "Installing dependencies..."
source .venv/bin/activate

# Use uv if available, otherwise pip
if command -v uv &> /dev/null; then
    uv pip install -e ".[modal]"
else
    pip install -e ".[modal]"
fi

echo "✓ Dependencies installed"

# Test database connection
echo ""
echo "Testing database connection..."
python3 -c "
import psycopg
conn = psycopg.connect('postgresql://knowledge:localdev@localhost:5433/knowledge_base')
result = conn.execute('SELECT COUNT(*) FROM documents').fetchone()
print(f'✓ Database connected ({result[0]} documents)')
conn.close()
"

# Modal setup reminder
echo ""
echo "=== Modal Setup (for GPU embedding) ==="
echo ""
echo "If you haven't already, run:"
echo "  modal setup          # Authenticate"
echo "  modal deploy modal_embedder.py  # Deploy embedder"
echo ""
echo "Or use --local flag to skip Modal and use CPU embedding."

# Claude Code config reminder
echo ""
echo "=== Claude Code Configuration ==="
echo ""
echo "Add to your Claude Code MCP config (~/.claude.json or similar):"
echo ""
cat << EOF
{
  "mcpServers": {
    "knowledge-base": {
      "command": "$(pwd)/.venv/bin/python",
      "args": ["$(pwd)/mcp_server.py"],
      "env": {
        "KB_DATABASE_URL": "postgresql://knowledge:localdev@localhost:5433/knowledge_base"
      }
    }
  }
}
EOF

echo ""
echo "=== Quick Start ==="
echo ""
echo "1. Ingest some documents:"
echo "   python ingest.py ~/notes"
echo ""
echo "2. Test the MCP server:"
echo "   python mcp_server.py"
echo ""
echo "3. (Optional) Watch for changes:"
echo "   python scripts/watch.py ~/notes"
echo ""
echo "Setup complete! 🎉"
