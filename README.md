# 📊 Plateforme NLP Multi-Agent — Analyse de Rapports

Plateforme locale d'analyse de rapports économiques/financiers avec architecture **multi-agent anti-hallucination**.

## ✨ Fonctionnalités

- **Upload** unitaire ou multiple de documents (PDF, TXT)
- **Extraction robuste** du contenu (texte + structure + tableaux)
- **Analyse NLP** pilotée par 5 agents spécialisés
- **Vérification factuelle** systématique (anti-hallucination)
- **Dashboard interactif** avec traçabilité complète
- **100% self-hosted** — aucune dépendance API externe

## 🏗️ Architecture Multi-Agent

```
📄 Document
    │
    ▼
┌──────────────┐
│ ParserAgent  │  → Extraction texte, structure, tables, métadonnées
└──────┬───────┘
       ▼
┌──────────────┐
│ IndexAgent   │  → Chunking, embeddings, indexation vectorielle (Qdrant)
└──────┬───────┘
       ▼
┌──────────────┐
│ AnalystAgent │  → Résumé, mots-clés, classification, claims (via LLM)
└──────┬───────┘
       ▼
┌───────────────┐
│ VerifierAgent │  → RAG retrieval + reranking + NLI (anti-hallucination)
└──────┬────────┘
       ▼
┌──────────────┐
│ EditorAgent  │  → Composition finale vérifiée + score de confiance
└──────────────┘
       │
       ▼
   📊 Résultat JSON + Dashboard
```

## 🛠️ Stack Technique

| Composant | Technologie |
|-----------|-------------|
| API Backend | FastAPI |
| Orchestration agents | LangGraph |
| LLM local | Ollama + Qwen2.5-7B-Instruct |
| Embeddings | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-base |
| Vérification factuelle | mDeBERTa-v3-base-mnli-xnli |
| Parsing documents | PyMuPDF |
| Base relationnelle | PostgreSQL |
| Base vectorielle | Qdrant |
| Queue tâches | Redis + Celery |
| Dashboard | Streamlit + Plotly |

## 📋 Prérequis

- **Python** 3.10+
- **Docker** & Docker Compose
- **Ollama** ([installer](https://ollama.ai))
- **GPU** NVIDIA recommandé (RTX 4060 ou supérieur)
- **RAM** 16 Go minimum (32 Go recommandé)

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <repo-url>
cd analyse_rapport_multi_agent_llm
```

### 2. Setup automatique

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 3. Ou setup manuel

```bash
# Environnement Python
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,monitoring]"

# Configuration
cp .env.example .env

# Services Docker (PostgreSQL, Qdrant, Redis)
docker-compose up -d

# Modèle LLM
ollama pull qwen2.5:7b-instruct-q4_K_M
```

## ▶️ Démarrage

### Option 1 : Script tout-en-un

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

### Option 2 : Démarrage manuel (3 terminaux)

**Terminal 1 — API FastAPI :**
```bash
source .venv/bin/activate
python -m app.main
```

**Terminal 2 — Worker Celery :**
```bash
source .venv/bin/activate
celery -A app.worker.tasks worker --loglevel=info --concurrency=1 --pool=solo
```

**Terminal 3 — Dashboard Streamlit :**
```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

### Accès

| Service | URL |
|---------|-----|
| API (Swagger) | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Qdrant UI | http://localhost:6333/dashboard |

## 📡 API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/upload` | Upload unitaire + analyse |
| `POST` | `/api/upload/batch` | Upload multiple + analyse |
| `GET` | `/api/analyses` | Historique des analyses |
| `GET` | `/api/analyses/{id}` | Détail d'une analyse |
| `GET` | `/api/stats` | Statistiques globales |
| `GET` | `/api/health` | État des services |

## 🛡️ Fiabilité (P0)

- Statuts détaillés par étape: `queued` → `parse` → `index` → `analyze` → `verify` → `edit` → `done` (ou `error`)
- Champ `failed_step` persisté en base pour diagnostiquer précisément l'étape en échec
- Retry par étape (pas de retry global Celery) avec backoff exponentiel borné
- Timeouts explicites par étape + annulation propre de la coroutine en timeout
- Dépendances critiques figées: `qdrant-client==1.17.0`, `transformers==4.57.6`, `torch==2.10.0`

## 📦 Structure du Projet

```
analyse_rapport_multi_agent_llm/
├── app/
│   ├── main.py                 # Application FastAPI
│   ├── config.py               # Configuration centralisée
│   ├── logging_config.py       # Logging structuré
│   ├── api/
│   │   └── routes.py           # Routes API REST
│   ├── agents/
│   │   ├── orchestrator.py     # Pipeline LangGraph
│   │   ├── parser_agent.py     # Agent 1 : Parsing
│   │   ├── index_agent.py      # Agent 2 : Indexation
│   │   ├── analyst_agent.py    # Agent 3 : Analyse
│   │   ├── verifier_agent.py   # Agent 4 : Vérification
│   │   └── editor_agent.py     # Agent 5 : Édition
│   ├── models/
│   │   ├── database.py         # Modèles SQLAlchemy
│   │   └── schemas.py          # Schémas Pydantic
│   ├── services/
│   │   ├── llm_service.py      # Interface Ollama
│   │   ├── embedding_service.py # Embeddings BGE-M3
│   │   ├── reranker_service.py # Reranking BGE
│   │   ├── nli_service.py      # NLI mDeBERTa
│   │   ├── vector_store.py     # Qdrant client
│   │   └── document_parser.py  # Parsing PDF/TXT
│   ├── db/
│   │   ├── session.py          # Session async SQLAlchemy
│   │   └── crud.py             # Opérations CRUD
│   └── worker/
│       └── tasks.py            # Tâches Celery
├── dashboard/
│   └── app.py                  # Interface Streamlit
├── tests/
│   ├── test_parser_agent.py
│   ├── test_index_agent.py
│   └── test_schemas.py
├── scripts/
│   ├── setup.sh
│   └── start.sh
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

## 🔒 Anti-Hallucination

Le pipeline garantit la fiabilité grâce à :

1. **Retrieval-Augmented Generation (RAG)** — chaque affirmation est ancrée dans le document source
2. **Reranking** — les passages les plus pertinents sont priorisés
3. **Natural Language Inference (NLI)** — vérification entailment/contradiction avec mDeBERTa
4. **Rejet automatique** — les affirmations non supportées sont exclues du résultat final
5. **Score de confiance** — chaque document reçoit un score de fiabilité global
6. **Traçabilité** — chaque claim est lié à sa source (page, chunk_id, citation)

## 📊 Format de Sortie

```json
{
  "doc_id": "uuid",
  "summary": "Résumé vérifié du document...",
  "section_summaries": [
    {
      "section_title": "Contexte macroéconomique",
      "summary": "Résumé de cette partie du document..."
    },
    {
      "section_title": "Perspectives sectorielles",
      "summary": "Résumé de cette partie du document..."
    }
  ],
  "keywords": ["économie", "croissance", "inflation"],
  "classification": {
    "label": "Politique monétaire",
    "score": 0.87
  },
  "claims": [
    {
      "text": "Le PIB a augmenté de 0.3% au T3 2025.",
      "status": "supported",
      "evidence": [
        {
          "page": 3,
          "chunk_id": "c_abc12345_3_0002",
          "quote": "La croissance du PIB s'est établie à 0.3%..."
        }
      ]
    }
  ],
  "confidence_global": 0.82,
  "processing_time_sec": 74.2
}
```

## ⚙️ Configuration Machine

Configuration testée :
- CPU : Intel Core i7-13620H
- GPU : NVIDIA RTX 4060 Laptop
- RAM : 16 Go

**Temps de traitement estimé** : 1-4 min par document selon la taille.

## 🧪 Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## 📜 Licence

MIT
