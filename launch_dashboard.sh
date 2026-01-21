#!/bin/bash
# Launch Atlas TNT Training Dashboard
# Usage: ./launch_dashboard.sh [port]

PORT=${1:-8501}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "🧠 Starting Atlas TNT Dashboard on port $PORT..."
echo "   Open: http://localhost:$PORT"
echo ""

.venv/bin/streamlit run scripts/dashboard.py \
    --server.port $PORT \
    --server.headless true \
    --theme.base dark \
    --browser.gatherUsageStats false
