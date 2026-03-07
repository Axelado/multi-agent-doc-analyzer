#!/usr/bin/env bash
# =============================================================
# Script de setup — Plateforme NLP Multi-Agent
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo " Plateforme NLP Multi-Agent — Setup"
echo "=========================================="

# --- 1. Vérifier les prérequis ---
echo ""
echo "[1/6] Vérification des prérequis..."

check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "❌ $1 non trouvé. Veuillez l'installer."
        return 1
    else
        echo "✅ $1 trouvé"
    fi
}

check_command python3
check_command docker
check_command docker-compose || check_command "docker-compose"

# --- 2. Environnement Python ---
echo ""
echo "[2/6] Configuration de l'environnement Python..."

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Environnement virtuel créé"
fi

source .venv/bin/activate
echo "✅ Environnement virtuel activé"

# --- 3. Installer les dépendances ---
echo ""
echo "[3/6] Installation des dépendances Python..."

pip install --upgrade pip setuptools wheel
pip install -e ".[dev,monitoring]"
echo "✅ Dépendances installées"

# --- 4. Configuration .env ---
echo ""
echo "[4/6] Configuration..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Fichier .env créé depuis .env.example"
else
    echo "✅ Fichier .env existant conservé"
fi

# --- 5. Services Docker ---
echo ""
echo "[5/6] Démarrage des services (PostgreSQL, Qdrant, Redis)..."

docker-compose up -d
echo "✅ Services Docker démarrés"

# Attendre que les services soient prêts
echo "  Attente de PostgreSQL..."
for i in $(seq 1 30); do
    if docker-compose exec -T postgres pg_isready -U nlp_user &>/dev/null; then
        echo "  ✅ PostgreSQL prêt"
        break
    fi
    sleep 1
done

echo "  Attente de Qdrant..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz &>/dev/null; then
        echo "  ✅ Qdrant prêt"
        break
    fi
    sleep 1
done

echo "  Attente de Redis..."
for i in $(seq 1 10); do
    if docker-compose exec -T redis redis-cli ping &>/dev/null; then
        echo "  ✅ Redis prêt"
        break
    fi
    sleep 1
done

# --- 6. Vérifier/Installer Ollama ---
echo ""
echo "[6/6] Vérification d'Ollama..."

if command -v ollama &>/dev/null; then
    echo "✅ Ollama trouvé"

    # Vérifier si le modèle est disponible
    if ! ollama list 2>/dev/null | grep -q "qwen2.5"; then
        echo "  📥 Téléchargement du modèle qwen2.5:7b-instruct-q4_K_M..."
        echo "  (Cela peut prendre plusieurs minutes)"
        ollama pull qwen2.5:7b-instruct-q4_K_M
        echo "  ✅ Modèle téléchargé"
    else
        echo "  ✅ Modèle qwen2.5 déjà disponible"
    fi
else
    echo "⚠️  Ollama non trouvé."
    echo "  Installez-le depuis: https://ollama.ai"
    echo "  Puis exécutez: ollama pull qwen2.5:7b-instruct-q4_K_M"
fi

# --- Résumé ---
echo ""
echo "=========================================="
echo " Setup terminé !"
echo "=========================================="
echo ""
echo " Pour démarrer la plateforme :"
echo ""
echo "   1. Activer l'environnement :"
echo "      source .venv/bin/activate"
echo ""
echo "   2. Démarrer l'API :"
echo "      python -m app.main"
echo ""
echo "   3. Démarrer le worker Celery :"
echo "      celery -A app.worker.tasks worker --loglevel=info --concurrency=1 --pool=solo"
echo ""
echo "   4. Démarrer le dashboard :"
echo "      streamlit run dashboard/app.py"
echo ""
echo "   5. Ouvrir dans le navigateur :"
echo "      - API :       http://localhost:8000/docs"
echo "      - Dashboard : http://localhost:8501"
echo ""
