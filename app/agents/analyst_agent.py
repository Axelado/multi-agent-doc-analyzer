"""Agent 3 — AnalystAgent : produire un brouillon analytique."""

from __future__ import annotations

import json
import re

import structlog

from app.config import get_settings
from app.models.schemas import Chunk, ParsedDocument
from app.services.llm_service import get_llm_service

logger = structlog.get_logger(__name__)


ANALYST_SYSTEM_PROMPT_TEMPLATE = """Tu es un analyste expert en rapports économiques et financiers.
Tu dois analyser le document fourni et produire un JSON structuré avec :
1. Un résumé détaillé, factuel et structuré (cible: {target_words} mots, minimum: {min_words}, maximum: {max_words})
2. Des mots-clés pertinents (5 à 15)
3. Une classification thématique avec score de confiance
4. Des affirmations clés extraites du document

RÈGLES STRICTES :
- Chaque affirmation doit être directement tirée du document
- Chaque affirmation doit référencer les chunk_ids sources
- Ne pas inventer d'informations
- Rester factuel et précis
- Répondre en français
- Le champ "summary" doit être rédigé en 4 à 6 paragraphes (contexte, faits/chiffres, tendances, points d'attention, conclusion)

Format de sortie OBLIGATOIRE (JSON valide) :
{{
    "summary": "résumé du document...",
    "keywords": ["mot1", "mot2", ...],
    "classification": {{
        "label": "catégorie thématique",
        "score": 0.85
    }},
    "claims": [
        {{
            "text": "affirmation extraite...",
            "source_chunk_ids": ["c_xxx_1_0001", "c_xxx_2_0003"]
        }}
    ]
}}"""


class AnalystAgent:
    """
    Rôle : produire un brouillon analytique (résumé + mots-clés + classification).
    Contrainte : chaque phrase de sortie doit référencer au moins un chunk_id candidat.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = get_llm_service()

    def _prepare_context(
        self, parsed: ParsedDocument, chunks: list[Chunk], max_chars: int | None = None
    ) -> str:
        """Préparer le contexte textuel pour le LLM."""
        max_chars = max_chars or self.settings.analyst_context_max_chars
        context_parts = []
        total_chars = 0

        for chunk in chunks:
            entry = f"[{chunk.chunk_id}] (page {chunk.page_number}): {chunk.text}"
            if total_chars + len(entry) > max_chars:
                break
            context_parts.append(entry)
            total_chars += len(entry)

        return "\n\n".join(context_parts)

    def _summary_word_count(self, summary: str) -> int:
        return len(re.findall(r"\b\w+\b", summary or ""))

    def _build_system_prompt(self) -> str:
        target_words = self.settings.analyst_summary_target_words
        min_words = self.settings.analyst_summary_min_words
        max_words = max(min_words + 80, target_words + 120)
        return ANALYST_SYSTEM_PROMPT_TEMPLATE.format(
            target_words=target_words,
            min_words=min_words,
            max_words=max_words,
        )

    def _clean_generated_summary(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:text|markdown|json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip().strip('"')

    async def _expand_summary_if_needed(
        self,
        *,
        parsed: ParsedDocument,
        context: str,
        result: dict,
    ) -> dict:
        min_words = self.settings.analyst_summary_min_words
        retries = max(0, self.settings.analyst_summary_retry_count)

        for _ in range(retries):
            current_summary = str(result.get("summary", "")).strip()
            current_words = self._summary_word_count(current_summary)

            if current_words >= min_words:
                break

            logger.info(
                "analyst_summary_too_short",
                doc_id=parsed.doc_id,
                summary_words=current_words,
                min_required=min_words,
            )

            target_words = self.settings.analyst_summary_target_words
            max_words = max(min_words + 80, target_words + 120)
            expansion_prompt = f"""Réécris le résumé ci-dessous pour qu'il soit plus complet.

Contraintes :
- Français
- Entre {min_words} et {max_words} mots
- 4 à 6 paragraphes
- Inclure faits/chiffres importants, tendances et points d'attention
- Ne rien inventer au-delà du contexte fourni
- Ne pas produire de JSON, uniquement le texte final du résumé

Document : {parsed.filename}
Type : {parsed.file_type}
Nombre de pages : {parsed.num_pages}

Résumé actuel :
{current_summary}

Contexte source :
{context}
"""

            expanded = self._clean_generated_summary(
                await self.llm.generate(expansion_prompt)
            )
            expanded_words = self._summary_word_count(expanded)

            if expanded_words > current_words:
                result["summary"] = expanded
                logger.info(
                    "analyst_summary_expanded",
                    doc_id=parsed.doc_id,
                    summary_words=expanded_words,
                )

        return result

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
        target_words = self.settings.analyst_summary_target_words
        min_words = self.settings.analyst_summary_min_words

        prompt = f"""Analyse le document suivant et produis le JSON structuré demandé.

    Exigences de résumé :
    - Le champ "summary" doit faire environ {target_words} mots
    - Longueur minimale obligatoire : {min_words} mots
    - Résumé en 4 à 6 paragraphes (pas de liste à puces)

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
            system_prompt=self._build_system_prompt(),
        )

        result = self._extract_json(response)

        # S'assurer que toutes les clés requises existent
        result.setdefault("summary", "")
        result.setdefault("keywords", [])
        result.setdefault("classification", {"label": "Non classifié", "score": 0.0})
        result.setdefault("claims", [])

        result = await self._expand_summary_if_needed(
            parsed=parsed,
            context=context,
            result=result,
        )

        logger.info(
            "analyst_agent_complete",
            doc_id=parsed.doc_id,
            num_claims=len(result.get("claims", [])),
            num_keywords=len(result.get("keywords", [])),
            summary_words=self._summary_word_count(str(result.get("summary", ""))),
        )

        return result
