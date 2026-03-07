"""Agent 5 — EditorAgent : composition de la version finale."""

from __future__ import annotations

import json
import re

import structlog

from app.models.schemas import Claim, ParsedDocument
from app.services.llm_service import get_llm_service

logger = structlog.get_logger(__name__)


EDITOR_SYSTEM_PROMPT = """Tu es un éditeur expert chargé de produire la version finale d'une analyse de document.

RÈGLES ABSOLUES :
- N'inclure QUE des informations validées (status = "supported")
- Exclure totalement les affirmations rejetées (status = "rejected")
- Le résumé doit être concis, précis et fidèle au document source
- Chaque affirmation clé doit avoir au moins une citation vérifiable
- Produire un JSON strictement valide

Format de sortie OBLIGATOIRE :
{
  "summary": "résumé final...",
  "keywords": ["mot1", "mot2", ...],
  "classification": {
    "label": "catégorie",
    "score": 0.85
  },
  "claims": [
    {
      "text": "affirmation vérifiée...",
      "status": "supported",
      "evidence": [
        {"page": 12, "chunk_id": "c_xxx", "quote": "citation source..."}
      ]
    }
  ],
  "confidence_global": 0.82
}"""


class EditorAgent:
    """
    Rôle : composer la version finale lisible et concise.
    Contrainte : n'inclure que des éléments validés par le VerifierAgent.

    Sortie finale :
    - résumé final,
    - mots-clés,
    - classification,
    - citations,
    - score de confiance global.
    """

    def __init__(self) -> None:
        self.llm = get_llm_service()

    def _calculate_confidence(self, verified: dict) -> float:
        """Calculer le score de confiance global."""
        claims = verified.get("claims", [])
        if not claims:
            return 0.5

        supported_count = sum(
            1 for c in claims if c.get("status") == "supported"
        )
        total = len(claims)

        claim_ratio = supported_count / total if total > 0 else 0.5

        # Pondérer par la confiance moyenne des claims supportés
        avg_confidence = 0.0
        supported_claims = [c for c in claims if c.get("status") == "supported"]
        if supported_claims:
            avg_confidence = sum(
                c.get("confidence", 0.5) for c in supported_claims
            ) / len(supported_claims)

        # Score composite
        confidence = 0.6 * claim_ratio + 0.4 * avg_confidence

        # Pénalité si le résumé a été contredit
        if not verified.get("summary_verified", True):
            confidence *= 0.8

        return round(min(max(confidence, 0.0), 1.0), 4)

    def _extract_json(self, response: str) -> dict:
        """Extraire le JSON de la réponse LLM."""
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    async def run(
        self, verified: dict, parsed: ParsedDocument
    ) -> dict:
        """Composer la version finale de l'analyse."""
        logger.info("editor_agent_start", doc_id=parsed.doc_id)

        # Filtrer : ne garder que les claims supportés
        supported_claims = verified.get("claims_supported", [])
        rejected_claims = verified.get("claims_rejected", [])

        # Calculer la confiance globale
        confidence = self._calculate_confidence(verified)

        # Si pas assez de claims supportés, mode simplifié
        if not supported_claims and not verified.get("summary", ""):
            logger.warning("no_supported_claims", doc_id=parsed.doc_id)
            return {
                "summary": "Analyse non concluante : aucune affirmation n'a pu être vérifiée.",
                "keywords": verified.get("keywords", []),
                "classification": verified.get("classification", {"label": "Non classifié", "score": 0.0}),
                "claims": [],
                "confidence_global": 0.0,
            }

        # Préparer le contexte pour le LLM
        claims_text = json.dumps(supported_claims, ensure_ascii=False, indent=2)
        rejected_text = json.dumps(
            [c.get("text", "") for c in rejected_claims],
            ensure_ascii=False,
        )

        prompt = f"""Voici les résultats de l'analyse vérifiée du document "{parsed.filename}":

RÉSUMÉ BROUILLON :
{verified.get('summary', 'Pas de résumé')}

AFFIRMATIONS VÉRIFIÉES (à inclure) :
{claims_text}

AFFIRMATIONS REJETÉES (à EXCLURE absolument) :
{rejected_text}

MOTS-CLÉS CANDIDATS :
{json.dumps(verified.get('keywords', []), ensure_ascii=False)}

CLASSIFICATION PROPOSÉE :
{json.dumps(verified.get('classification', {}), ensure_ascii=False)}

Score de confiance calculé : {confidence}

Produis maintenant la version FINALE en JSON avec uniquement les informations vérifiées.
Le résumé doit être réécrit de manière claire, sans inclure d'affirmations rejetées."""

        response = await self.llm.generate_structured(
            prompt=prompt,
            system_prompt=EDITOR_SYSTEM_PROMPT,
        )

        result = self._extract_json(response)

        # Garantir la structure
        final = {
            "summary": result.get("summary", verified.get("summary", "")),
            "keywords": result.get("keywords", verified.get("keywords", [])),
            "classification": result.get(
                "classification",
                verified.get("classification", {"label": "Non classifié", "score": 0.0}),
            ),
            "claims": result.get("claims", supported_claims),
            "confidence_global": confidence,
        }

        # S'assurer que les claims ont le bon format
        formatted_claims = []
        for claim in final["claims"]:
            if isinstance(claim, dict):
                formatted_claims.append({
                    "text": claim.get("text", ""),
                    "status": claim.get("status", "supported"),
                    "evidence": claim.get("evidence", []),
                })
            else:
                formatted_claims.append({
                    "text": str(claim),
                    "status": "unverified",
                    "evidence": [],
                })
        final["claims"] = formatted_claims

        logger.info(
            "editor_agent_complete",
            doc_id=parsed.doc_id,
            confidence=confidence,
            num_final_claims=len(formatted_claims),
        )

        return final
