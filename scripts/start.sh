#!/usr/bin/env bash
# =============================================================
# Script de démarrage — Plateforme NLP Multi-Agent
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=========================================="
echo " Démarrage de la plateforme"
echo "=========================================="

# Vérifier les services Docker
echo "[1/4] Vérification des services Docker..."
docker-compose up -d
echo "  ✅ Services Docker actifs"

# Vérifier Ollama
echo "[2/4] Vérification d'Ollama..."
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "  ✅ Ollama actif"
else
    echo "  ⚠️  Ollama non accessible. Démarrez-le avec: ollama serve"
fi

# Démarrer le worker Celery en arrière-plan
echo "[3/4] Démarrage du worker Celery..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
celery -A app.worker.tasks worker --loglevel=info --concurrency=1 --pool=solo &
CELERY_PID=$!
echo "  ✅ Worker Celery démarré (PID: $CELERY_PID)"

# Démarrer le dashboard Streamlit en arrière-plan
echo "[4/4] Démarrage du dashboard Streamlit..."
streamlit run dashboard/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!
echo "  ✅ Dashboard démarré (PID: $STREAMLIT_PID)"

echo ""
echo "=========================================="
echo " Démarrage de l'API FastAPI..."
echo "=========================================="
echo ""
echo " - API :       http://localhost:8000/docs"
echo " - Dashboard : http://localhost:8501"
echo ""

# Démarrer l'API en premier plan
python -m app.main

# Cleanup si l'API est arrêtée
kill $CELERY_PID 2>/dev/null || true
kill $STREAMLIT_PID 2>/dev/null || true
