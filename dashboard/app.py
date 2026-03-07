"""Dashboard Streamlit — Interface utilisateur interactive."""

import json
import time
from datetime import datetime

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================
# Configuration
# =============================================================

API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="📊 Plateforme NLP Multi-Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# Fonctions utilitaires API
# =============================================================


def api_get(endpoint: str) -> dict | None:
    """Requête GET vers l'API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{API_BASE_URL}{endpoint}")
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Erreur API: {resp.status_code} — {resp.text}")
                return None
    except httpx.ConnectError:
        st.error("❌ Impossible de se connecter à l'API. Vérifiez que le serveur est lancé.")
        return None
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return None


def api_upload(files: list) -> dict | None:
    """Upload de fichiers vers l'API."""
    try:
        with httpx.Client(timeout=120.0) as client:
            if len(files) == 1:
                resp = client.post(
                    f"{API_BASE_URL}/upload",
                    files={"file": (files[0].name, files[0].getvalue(), "application/octet-stream")},
                )
            else:
                multi_files = [
                    ("files", (f.name, f.getvalue(), "application/octet-stream"))
                    for f in files
                ]
                resp = client.post(f"{API_BASE_URL}/upload/batch", files=multi_files)

            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Erreur upload: {resp.status_code} — {resp.text}")
                return None
    except Exception as e:
        st.error(f"Erreur upload: {str(e)}")
        return None


# =============================================================
# Sidebar
# =============================================================

with st.sidebar:
    st.title("📊 NLP Multi-Agent")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "📤 Upload", "📋 Analyses", "📈 Statistiques", "🔍 Détail"],
        index=0,
    )

    st.markdown("---")

    # Health check
    if st.button("🔄 Vérifier les services"):
        health = api_get("/health")
        if health:
            status = health.get("status", "unknown")
            if status == "ok":
                st.success("✅ Tous les services fonctionnent")
            else:
                st.warning(f"⚠️ Statut: {status}")

            for service, state in health.get("services", {}).items():
                icon = "✅" if state == "ok" else "❌"
                st.write(f"{icon} {service}: {state}")

    st.markdown("---")
    st.caption("v0.1.0 — Multi-Agent NLP Platform")


# =============================================================
# Pages
# =============================================================

if page == "🏠 Accueil":
    st.title("🏠 Plateforme NLP Multi-Agent")
    st.markdown("""
    ### Bienvenue !

    Cette plateforme analyse vos rapports économiques et financiers grâce à une
    architecture **multi-agent anti-hallucination**.

    #### 🔄 Pipeline d'analyse

    | Étape | Agent | Rôle |
    |-------|-------|------|
    | 1 | **ParserAgent** | Extraction du contenu (texte, structure, tables) |
    | 2 | **IndexAgent** | Indexation sémantique (embeddings + Qdrant) |
    | 3 | **AnalystAgent** | Analyse (résumé, mots-clés, classification) |
    | 4 | **VerifierAgent** | Vérification factuelle (NLI anti-hallucination) |
    | 5 | **EditorAgent** | Composition finale vérifiée |

    #### 📌 Critères anti-hallucination
    - Aucune phrase finale sans preuve (page + chunk_id)
    - Rejet automatique des assertions contradictoires
    - Score de confiance explicite par document
    - Traçabilité complète des sources
    """)

    # Quick stats
    stats = api_get("/stats")
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Documents", stats.get("total_documents", 0))
        col2.metric("✅ Analysés", stats.get("completed", 0))
        col3.metric("⏳ En cours", stats.get("processing", 0) + stats.get("pending", 0))
        col4.metric(
            "🎯 Confiance moy.",
            f"{stats['avg_confidence']:.0%}" if stats.get("avg_confidence") else "N/A",
        )


elif page == "📤 Upload":
    st.title("📤 Upload de documents")
    st.markdown("Uploadez un ou plusieurs fichiers **PDF** ou **TXT** pour analyse.")

    uploaded_files = st.file_uploader(
        "Choisir les fichiers",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} fichier(s) sélectionné(s)")

        for f in uploaded_files:
            size_mb = len(f.getvalue()) / (1024 * 1024)
            st.write(f"  • **{f.name}** — {size_mb:.2f} MB")

        if st.button("🚀 Lancer l'analyse", type="primary"):
            with st.spinner("Upload et lancement de l'analyse..."):
                result = api_upload(uploaded_files)

            if result:
                st.success("✅ Document(s) uploadé(s) avec succès !")

                if "uploads" in result:
                    for upload in result["uploads"]:
                        st.write(
                            f"  • **{upload['filename']}** — ID: `{upload['doc_id']}` — "
                            f"Statut: {upload['status']}"
                        )
                else:
                    st.write(
                        f"  • **{result['filename']}** — ID: `{result['doc_id']}` — "
                        f"Statut: {result['status']}"
                    )

                st.info("⏳ L'analyse est en cours. Consultez la page 'Analyses' pour le suivi.")


elif page == "📋 Analyses":
    st.title("📋 Historique des analyses")

    data = api_get("/analyses?limit=100")
    if data and data.get("documents"):
        docs = data["documents"]

        # Tableau récapitulatif
        table_data = []
        for doc in docs:
            status_icon = {
                "completed": "✅",
                "processing": "⏳",
                "pending": "🕐",
                "error": "❌",
            }.get(doc["status"], "❓")
            num_pages = doc.get("num_pages")

            table_data.append({
                "Statut": f"{status_icon} {doc['status']}",
                "Fichier": doc["filename"],
                "Type": doc["file_type"].upper(),
                "Pages": str(num_pages) if num_pages is not None else "-",
                "Confiance": f"{doc['confidence_global']:.0%}" if doc.get("confidence_global") else "-",
                "Temps (s)": f"{doc['processing_time_sec']:.1f}" if doc.get("processing_time_sec") else "-",
                "Date": doc["created_at"][:19],
                "ID": str(doc["id"])[:8],
            })

        st.dataframe(table_data, width="stretch", hide_index=True)

        # Auto-refresh
        if any(d["status"] in ("pending", "processing") for d in docs):
            st.info("⏳ Des analyses sont en cours. Rafraîchissement automatique dans 10s...")
            time.sleep(10)
            st.rerun()
    else:
        st.info("Aucune analyse disponible. Uploadez un document pour commencer.")


elif page == "📈 Statistiques":
    st.title("📈 Statistiques globales")

    stats = api_get("/stats")
    if stats:
        col1, col2 = st.columns(2)

        with col1:
            # Graphique de répartition des statuts
            status_data = {
                "Complété": stats.get("completed", 0),
                "En cours": stats.get("processing", 0),
                "En attente": stats.get("pending", 0),
                "Erreur": stats.get("failed", 0),
            }
            fig_pie = px.pie(
                names=list(status_data.keys()),
                values=list(status_data.values()),
                title="Répartition par statut",
                color_discrete_sequence=["#2ecc71", "#f39c12", "#3498db", "#e74c3c"],
            )
            st.plotly_chart(fig_pie, width="stretch")

        with col2:
            st.metric("📄 Total documents", stats.get("total_documents", 0))
            st.metric(
                "🎯 Confiance moyenne",
                f"{stats['avg_confidence']:.2%}" if stats.get("avg_confidence") else "N/A",
            )
            st.metric(
                "⏱️ Temps moyen",
                f"{stats['avg_processing_time']:.1f}s" if stats.get("avg_processing_time") else "N/A",
            )

        # Historique des analyses (confiance au fil du temps)
        data = api_get("/analyses?limit=50")
        if data and data.get("documents"):
            completed = [
                d for d in data["documents"]
                if d["status"] == "completed" and d.get("confidence_global") is not None
            ]
            if completed:
                fig_hist = px.bar(
                    x=[d["filename"][:20] for d in completed],
                    y=[d["confidence_global"] for d in completed],
                    title="Score de confiance par document",
                    labels={"x": "Document", "y": "Confiance"},
                    color=[d["confidence_global"] for d in completed],
                    color_continuous_scale="RdYlGn",
                )
                fig_hist.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig_hist, width="stretch")
    else:
        st.warning("Impossible de charger les statistiques.")


elif page == "🔍 Détail":
    st.title("🔍 Détail d'une analyse")

    # Sélecteur de document
    data = api_get("/analyses?limit=100")
    if data and data.get("documents"):
        docs = data["documents"]
        options = {f"{d['filename']} ({str(d['id'])[:8]})": d["id"] for d in docs}

        selected = st.selectbox("Choisir un document", list(options.keys()))

        if selected:
            doc_id = options[selected]
            doc = api_get(f"/analyses/{doc_id}")

            if doc:
                # En-tête
                col1, col2, col3 = st.columns(3)
                col1.metric("Statut", doc["status"])
                col2.metric(
                    "Confiance",
                    f"{doc['confidence_global']:.0%}" if doc.get("confidence_global") else "N/A",
                )
                col3.metric(
                    "Temps",
                    f"{doc['processing_time_sec']:.1f}s" if doc.get("processing_time_sec") else "N/A",
                )

                st.markdown("---")

                if doc["status"] == "completed":
                    # Résumé
                    st.subheader("📝 Résumé")
                    st.write(doc.get("summary", "Pas de résumé disponible"))

                    # Mots-clés
                    st.subheader("🏷️ Mots-clés")
                    keywords = doc.get("keywords", [])
                    if keywords:
                        st.write(" • ".join([f"`{kw}`" for kw in keywords]))

                    # Classification
                    st.subheader("📂 Classification")
                    classification = doc.get("classification", {})
                    if classification:
                        st.write(
                            f"**{classification.get('label', 'N/A')}** "
                            f"(score: {classification.get('score', 0):.2f})"
                        )

                    # Affirmations vérifiées
                    st.subheader("✅ Affirmations vérifiées")
                    claims = doc.get("claims", [])
                    if claims:
                        for i, claim in enumerate(claims, 1):
                            status_icon = {
                                "supported": "✅",
                                "rejected": "❌",
                                "unverified": "❓",
                            }.get(claim.get("status", ""), "❓")

                            with st.expander(
                                f"{status_icon} Affirmation {i}: {claim.get('text', '')[:80]}..."
                            ):
                                st.write(f"**Texte:** {claim.get('text', '')}")
                                st.write(f"**Statut:** {claim.get('status', 'N/A')}")

                                evidence = claim.get("evidence", [])
                                if evidence:
                                    st.write("**Preuves:**")
                                    for ev in evidence:
                                        st.info(
                                            f"📄 Page {ev.get('page', '?')} | "
                                            f"Chunk: `{ev.get('chunk_id', '?')}` | "
                                            f"Score: {ev.get('score', 0):.2f}\n\n"
                                            f"> {ev.get('quote', 'N/A')}"
                                        )
                    else:
                        st.info("Aucune affirmation extraite.")

                    # JSON brut
                    with st.expander("📦 Données JSON brutes"):
                        st.json(doc)

                elif doc["status"] == "error":
                    st.error(f"❌ Erreur: {doc.get('error_message', 'Erreur inconnue')}")

                elif doc["status"] in ("pending", "processing"):
                    st.info("⏳ Analyse en cours...")
                    time.sleep(5)
                    st.rerun()
    else:
        st.info("Aucun document disponible.")
