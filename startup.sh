#!/bin/bash
set -e

echo "=== Focentra ML Service Startup ==="
mkdir -p app/models/personal
echo "Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT