#!/bin/sh
# ============================================================
# pull-models.sh — Download Ollama models on first start
# ============================================================
# This script runs inside the ollama-pull container.
# It downloads the models that DocuMind needs into the
# shared ollama-data volume.
#
# Models downloaded:
#   llama3.2         — chat model for answering questions (~2GB)
#   nomic-embed-text — embedding model for vector search (~274MB)
#
# To add more models: add `ollama pull <model>` lines below.
# To use a bigger/smarter model: replace llama3.2 with llama3.1:8b
# or mistral — just update CHAT_MODEL in docker-compose.yml too.
# ============================================================

set -e   # exit immediately if any command fails

OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}

echo "============================================"
echo "DocuMind — Pulling required Ollama models"
echo "Ollama host: $OLLAMA_HOST"
echo "============================================"

# Wait until Ollama API is ready
echo "Waiting for Ollama to be ready..."
until curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
    echo "  Ollama not ready yet — retrying in 2s..."
    sleep 2
done
echo "Ollama is ready."

# Pull chat model
echo ""
echo "--- Pulling chat model: llama3.2 ---"
echo "Size: ~2.0 GB | This may take several minutes on first run"
ollama pull llama3.2

# Pull embedding model
echo ""
echo "--- Pulling embedding model: nomic-embed-text ---"
echo "Size: ~274 MB"
ollama pull nomic-embed-text

echo ""
echo "============================================"
echo "All models ready. DocuMind can now start."
echo "============================================"