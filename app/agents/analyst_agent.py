"""Agent 3 — AnalystAgent : produire un brouillon analytique."""

from __future__ import annotations

import json
import re

import structlog

from app.models.schemas import Chunk, ParsedDocument
from app.services.llm_service import get_llm_service

logger = structlog.get_logger(__name__)


ANALYST_SYSTEM_PROMPT = """Tu es un analyste expert en rapports économiques et financiers.
Tu dois analyser le document fourni et produire un JSON structuré avec :
1. Un résumé concis et factuel (max 500 mots)
2. Des mots-clés pertinents (5 à 15)
3. Une classification thématique avec score de confiance
4. Des affirmations clés extraites du document

RÈGLES STRICTES :
- Chaque affirmation doit être directement tirée du document
- Chaque affirmation doit référencer les chunk_ids sources
- Ne pas inventer d'informations
- Rester factuel et précis
- Répondre en français

Format de sortie OBLIGATOIRE (JSON valide) :
{
  "summary": "résumé du document...",
  "keywords": ["mot1", "mot2", ...],
  "classification": {
    "label": "catégorie thématique",
    "score": 0.85
  },
  "claims": [
    {
      "text": "affirmation extraite...",
      "source_chunk_ids": ["c_xxx_1_0001", "c_xxx_2_0003"]
    }
  ]
}"""


class AnalystAgent:
    """
    Rôle : produire un brouillon analytique (résumé + mots-clés + classification).
    Contrainte : chaque phrase de sortie doit référencer au moins un chunk_id candidat.
    """

    def __init__(self) -> None:
        self.llm = get_llm_service()

    def _prepare_context(
        self, parsed: ParsedDocument, chunks: list[Chunk], max_chars: int = 8000
    ) -> str:
        """Préparer le contexte textuel pour le LLM."""
        context_parts = []
        total_chars = 0

        for chunk in chunks:
            entry = f"[{chunk.chunk_id}] (page {chunk.page_number}): {chunk.text}"
            if total_chars + len(entry) > max_chars:
                break
            context_parts.append(entry)
            total_chars += len(entry)

        return "\n\n".join(context_parts)

    def _extract_json(self, response: str) -> dict:
        """Extraire le JSON de la réponse LLM."""
        # Chercher un bloc JSON dans la réponse
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Essai direct
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", response_preview=response[:200])
            return {
                "summary": response[:500],
                "keywords": [],
                "classification": {"label": "Non classifié", "score": 0.0},
                "claims": [],
            }

    async def run(
        self, parsed: ParsedDocument, chunks: list[Chunk]
    ) -> dict:
        """Produire l'analyse brouillon."""
        logger.info("analyst_agent_start", doc_id=parsed.doc_id)

        context = self._prepare_context(parsed, chunks)

        prompt = f"""Analyse le document suivant et produis le JSON structuré demandé.

Document : {parsed.filename}
Type : {parsed.file_type}
Nombre de pages : {parsed.num_pages}
Langue : {parsed.language or 'non détectée'}

--- CONTENU DU DOCUMENT ---
{context}
--- FIN DU CONTENU ---

Produis maintenant le JSON d'analyse complet."""

        response = await self.llm.generate_structured(
            prompt=prompt,
            system_prompt=ANALYST_SYSTEM_PROMPT,
        )

        result = self._extract_json(response)

        # S'assurer que toutes les clés requises existent
        result.setdefault("summary", "")
        result.setdefault("keywords", [])
        result.setdefault("classification", {"label": "Non classifié", "score": 0.0})
        result.setdefault("claims", [])

        logger.info(
            "analyst_agent_complete",
            doc_id=parsed.doc_id,
            num_claims=len(result.get("claims", [])),
            num_keywords=len(result.get("keywords", [])),
        )

        return result
