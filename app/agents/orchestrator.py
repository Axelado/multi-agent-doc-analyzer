"""Orchestrateur multi-agent avec LangGraph."""

from __future__ import annotations

import time
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from app.agents.analyst_agent import AnalystAgent
from app.agents.editor_agent import EditorAgent
from app.agents.index_agent import IndexAgent
from app.agents.parser_agent import ParserAgent
from app.agents.verifier_agent import VerifierAgent
from app.models.schemas import Chunk, ParsedDocument

logger = structlog.get_logger(__name__)


class PipelineState(TypedDict, total=False):
    """État partagé entre les agents du pipeline."""

    # Entrées
    file_path: str
    doc_id: str

    # Sorties intermédiaires
    parsed_document: ParsedDocument | None
    chunks: list[Chunk]
    draft_analysis: dict
    verified_analysis: dict

    # Sortie finale
    final_result: dict

    # Métadonnées
    start_time: float
    error: str | None
    current_step: str


class AgentOrchestrator:
    """Pipeline multi-agent orchestré par LangGraph."""

    def __init__(self) -> None:
        self.parser_agent = ParserAgent()
        self.index_agent = IndexAgent()
        self.analyst_agent = AnalystAgent()
        self.verifier_agent = VerifierAgent()
        self.editor_agent = EditorAgent()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construire le graphe LangGraph."""
        workflow = StateGraph(PipelineState)

        # Ajouter les nœuds
        workflow.add_node("parse", self._parse_step)
        workflow.add_node("index", self._index_step)
        workflow.add_node("analyze", self._analyze_step)
        workflow.add_node("verify", self._verify_step)
        workflow.add_node("edit", self._edit_step)

        # Définir le flux séquentiel
        workflow.set_entry_point("parse")
        workflow.add_edge("parse", "index")
        workflow.add_edge("index", "analyze")
        workflow.add_edge("analyze", "verify")
        workflow.add_edge("verify", "edit")
        workflow.add_edge("edit", END)

        return workflow.compile()

    async def _parse_step(self, state: PipelineState) -> dict:
        """Étape 1 : Parsing du document."""
        logger.info("pipeline_step", step="parse", doc_id=state["doc_id"])
        try:
            parsed = await self.parser_agent.run(
                file_path=state["file_path"],
                doc_id=state["doc_id"],
            )
            return {
                "parsed_document": parsed,
                "current_step": "parse_complete",
            }
        except Exception as e:
            logger.error("parse_step_error", error=str(e))
            return {"error": f"Erreur parsing: {str(e)}"}

    async def _index_step(self, state: PipelineState) -> dict:
        """Étape 2 : Indexation sémantique."""
        logger.info("pipeline_step", step="index", doc_id=state["doc_id"])
        if state.get("error"):
            return {}

        try:
            parsed = state["parsed_document"]
            chunks = await self.index_agent.run(parsed)
            return {
                "chunks": chunks,
                "current_step": "index_complete",
            }
        except Exception as e:
            logger.error("index_step_error", error=str(e))
            return {"error": f"Erreur indexation: {str(e)}"}

    async def _analyze_step(self, state: PipelineState) -> dict:
        """Étape 3 : Analyse (brouillon)."""
        logger.info("pipeline_step", step="analyze", doc_id=state["doc_id"])
        if state.get("error"):
            return {}

        try:
            parsed = state["parsed_document"]
            chunks = state["chunks"]
            draft = await self.analyst_agent.run(parsed, chunks)
            return {
                "draft_analysis": draft,
                "current_step": "analyze_complete",
            }
        except Exception as e:
            logger.error("analyze_step_error", error=str(e))
            return {"error": f"Erreur analyse: {str(e)}"}

    async def _verify_step(self, state: PipelineState) -> dict:
        """Étape 4 : Vérification factuelle."""
        logger.info("pipeline_step", step="verify", doc_id=state["doc_id"])
        if state.get("error"):
            return {}

        try:
            draft = state["draft_analysis"]
            chunks = state["chunks"]
            verified = await self.verifier_agent.run(
                draft_analysis=draft,
                doc_id=state["doc_id"],
                chunks=chunks,
            )
            return {
                "verified_analysis": verified,
                "current_step": "verify_complete",
            }
        except Exception as e:
            logger.error("verify_step_error", error=str(e))
            return {"error": f"Erreur vérification: {str(e)}"}

    async def _edit_step(self, state: PipelineState) -> dict:
        """Étape 5 : Édition finale."""
        logger.info("pipeline_step", step="edit", doc_id=state["doc_id"])
        if state.get("error"):
            return {}

        try:
            verified = state["verified_analysis"]
            parsed = state["parsed_document"]
            final = await self.editor_agent.run(verified, parsed)

            # Ajouter le temps de traitement
            elapsed = time.time() - state.get("start_time", time.time())
            final["processing_time_sec"] = round(elapsed, 2)
            final["doc_id"] = state["doc_id"]

            return {
                "final_result": final,
                "current_step": "complete",
            }
        except Exception as e:
            logger.error("edit_step_error", error=str(e))
            return {"error": f"Erreur édition: {str(e)}"}

    async def run(self, file_path: str, doc_id: str) -> dict:
        """Exécuter le pipeline complet d'analyse."""
        logger.info("pipeline_start", doc_id=doc_id, file_path=file_path)
        start_time = time.time()

        initial_state: PipelineState = {
            "file_path": file_path,
            "doc_id": doc_id,
            "parsed_document": None,
            "chunks": [],
            "draft_analysis": {},
            "verified_analysis": {},
            "final_result": {},
            "start_time": start_time,
            "error": None,
            "current_step": "starting",
        }

        # Exécuter le graphe
        final_state = await self._graph.ainvoke(initial_state)

        elapsed = time.time() - start_time

        if final_state.get("error"):
            logger.error(
                "pipeline_failed",
                doc_id=doc_id,
                error=final_state["error"],
                elapsed=round(elapsed, 2),
            )
            raise RuntimeError(final_state["error"])

        result = final_state.get("final_result", {})
        parsed = final_state.get("parsed_document")

        # Enrichir le résultat avec les métadonnées
        result["doc_id"] = doc_id
        result["processing_time_sec"] = round(elapsed, 2)

        if parsed:
            result["_metadata"] = {
                "num_pages": parsed.num_pages,
                "language": parsed.language,
                "parsed_metadata": parsed.metadata,
            }

        logger.info(
            "pipeline_complete",
            doc_id=doc_id,
            elapsed=round(elapsed, 2),
            confidence=result.get("confidence_global"),
        )

        return result


# Singleton
_orchestrator: AgentOrchestrator | None = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
