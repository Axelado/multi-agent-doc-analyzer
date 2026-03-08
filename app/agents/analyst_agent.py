"""Agent 3 — AnalystAgent : produire un brouillon analytique."""

from __future__ import annotations

import json
import re

import structlog

from app.config import get_settings
from app.models.schemas import Chunk, ParsedDocument
from app.services.llm_service import get_llm_service

logger = structlog.get_logger(__name__)


SUMMARY_SECTION_TITLES = (
    "## 1) Contexte",
    "## 2) Points clés",
    "## 3) Chiffres et faits",
    "## 4) Risques et limites",
    "## 5) Conclusion",
)


ANALYST_SYSTEM_PROMPT_TEMPLATE = """Tu es un analyste expert en rapports économiques et financiers.
Tu dois analyser le document fourni et produire un JSON structuré avec :
1. Un résumé détaillé, factuel et structuré (cible: {target_words} mots, minimum: {min_words}, maximum: {max_words})
2. Des résumés par section dans "section_summaries" (un résumé par partie du document)
3. Des mots-clés pertinents (5 à 15)
4. Une classification thématique avec score de confiance
5. Des affirmations clés extraites du document

RÈGLES STRICTES :
- Chaque affirmation doit être directement tirée du document
- Chaque affirmation doit référencer les chunk_ids sources
- Ne pas inventer d'informations
- Rester factuel et précis
- Répondre en français
- Le champ "summary" doit être structuré avec EXACTEMENT ces titres :
{summary_structure}
- Chaque section contient 2 à 5 phrases factuelles
- Le champ "section_summaries" doit être une liste d'objets
- Chaque objet doit avoir les clés "section_title" et "summary"
- Si des sections détectées sont fournies, respecter leurs titres

Format de sortie OBLIGATOIRE (JSON valide) :
{{
    "summary": "résumé du document...",
    "section_summaries": [
        {{
            "section_title": "Titre de section",
            "summary": "Résumé de la section..."
        }}
    ],
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

    def _extract_section_titles(self, parsed: ParsedDocument) -> list[str]:
        """Extraire les titres de sections détectés dans le document."""
        max_sections = max(1, self.settings.analyst_section_summary_max)
        titles: list[str] = []
        seen: set[str] = set()

        for block in parsed.structured_blocks:
            if block.block_type != "title":
                continue

            cleaned = " ".join((block.content or "").split()).strip(" :-")
            if len(cleaned) < 3:
                continue

            normalized = cleaned.lower()
            if normalized in seen:
                continue

            seen.add(normalized)
            titles.append(cleaned)

            if len(titles) >= max_sections:
                break

        return titles

    def _normalize_section_summaries(
        self,
        section_summaries: object,
        section_titles: list[str],
    ) -> list[dict[str, str]]:
        """Normaliser section_summaries vers [{section_title, summary}, ...]."""
        if not isinstance(section_summaries, list):
            return []

        normalized_entries: list[dict[str, str]] = []
        for idx, item in enumerate(section_summaries, start=1):
            if isinstance(item, dict):
                raw_title = (
                    item.get("section_title")
                    or item.get("title")
                    or item.get("section")
                    or ""
                )
                raw_summary = item.get("summary") or item.get("text") or ""
            else:
                raw_title = ""
                raw_summary = str(item)

            title = " ".join(str(raw_title).split()).strip()
            summary = " ".join(str(raw_summary).split()).strip()

            if not summary:
                continue

            if not title:
                if idx - 1 < len(section_titles):
                    title = section_titles[idx - 1]
                else:
                    title = f"Partie {idx}"

            normalized_entries.append(
                {
                    "section_title": title,
                    "summary": summary,
                }
            )

            if len(normalized_entries) >= max(1, self.settings.analyst_section_summary_max):
                break

        return normalized_entries

    def _needs_section_summaries_generation(
        self,
        section_summaries: list[dict[str, str]],
        section_titles: list[str],
    ) -> bool:
        if not section_summaries:
            return True

        if not section_titles:
            return False

        produced_titles = {entry["section_title"].strip().lower() for entry in section_summaries}
        required_titles = {title.strip().lower() for title in section_titles}

        # On exige au moins 70% de couverture des sections détectées
        if not required_titles:
            return False
        covered = len(produced_titles.intersection(required_titles))
        ratio = covered / len(required_titles)
        return ratio < 0.7

    def _summary_word_count(self, summary: str) -> int:
        return len(re.findall(r"\b\w+\b", summary or ""))

    def _build_system_prompt(self) -> str:
        target_words = self.settings.analyst_summary_target_words
        min_words = self.settings.analyst_summary_min_words
        max_words = max(min_words + 80, target_words + 120)
        summary_structure = "\n".join(SUMMARY_SECTION_TITLES)
        return ANALYST_SYSTEM_PROMPT_TEMPLATE.format(
            target_words=target_words,
            min_words=min_words,
            max_words=max_words,
            summary_structure=summary_structure,
        )

    def _is_structured_summary(self, summary: str) -> bool:
        normalized = (summary or "").lower()
        return all(title.lower() in normalized for title in SUMMARY_SECTION_TITLES)

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
            is_structured = self._is_structured_summary(current_summary)
            needs_length = current_words < min_words
            needs_structure = not is_structured

            if not needs_length and not needs_structure:
                break

            logger.info(
                "analyst_summary_refine_needed",
                doc_id=parsed.doc_id,
                summary_words=current_words,
                min_required=min_words,
                needs_length=needs_length,
                needs_structure=needs_structure,
            )

            target_words = self.settings.analyst_summary_target_words
            max_words = max(min_words + 80, target_words + 120)
            summary_structure = "\n".join(SUMMARY_SECTION_TITLES)
            expansion_prompt = f"""Réécris le résumé ci-dessous pour qu'il soit plus complet et structuré.

Contraintes :
- Français
- Entre {min_words} et {max_words} mots
- Structure OBLIGATOIRE avec exactement ces titres :
{summary_structure}
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
            expanded_is_structured = self._is_structured_summary(expanded)

            if expanded_words > current_words or (not is_structured and expanded_is_structured):
                result["summary"] = expanded
                logger.info(
                    "analyst_summary_expanded",
                    doc_id=parsed.doc_id,
                    summary_words=expanded_words,
                    structured=expanded_is_structured,
                )

        return result

    async def _ensure_section_summaries(
        self,
        *,
        parsed: ParsedDocument,
        context: str,
        result: dict,
        section_titles: list[str],
    ) -> dict:
        current = self._normalize_section_summaries(
            result.get("section_summaries", []),
            section_titles,
        )
        if not self._needs_section_summaries_generation(current, section_titles):
            result["section_summaries"] = current
            return result

        logger.info(
            "analyst_section_summaries_refine_needed",
            doc_id=parsed.doc_id,
            current_sections=len(current),
            detected_sections=len(section_titles),
        )

        sections_hint = (
            "\n".join(f"- {title}" for title in section_titles)
            if section_titles
            else "(Aucun titre explicite détecté : infère des parties logiques du document)"
        )

        prompt = f"""Tu dois produire des résumés par section pour le document ci-dessous.

Contraintes :
- Répondre en JSON VALIDE uniquement (sans markdown)
- Format exact attendu :
[
  {{"section_title": "Titre", "summary": "Résumé"}}
]
- Une entrée par partie du document
- Si des sections détectées sont fournies, reprendre exactement leurs titres
- Chaque résumé de section: 2 à 5 phrases factuelles
- Ne rien inventer au-delà du contexte fourni

Sections détectées :
{sections_hint}

Document : {parsed.filename}
Type : {parsed.file_type}
Nombre de pages : {parsed.num_pages}

Contexte source :
{context}
"""

        raw_response = await self.llm.generate(prompt)
        candidate: object
        array_match = re.search(r"\[[\s\S]*\]", raw_response)
        if array_match:
            try:
                candidate = json.loads(array_match.group())
            except json.JSONDecodeError:
                candidate = []
        else:
            try:
                candidate = json.loads(raw_response)
            except json.JSONDecodeError:
                candidate = []

        generated = self._normalize_section_summaries(candidate, section_titles)
        result["section_summaries"] = generated if generated else current
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
        section_titles = self._extract_section_titles(parsed)
        target_words = self.settings.analyst_summary_target_words
        min_words = self.settings.analyst_summary_min_words
        sections_hint = (
            "\n".join(f"- {title}" for title in section_titles)
            if section_titles
            else "(Aucun titre explicite détecté : déduis des parties logiques du document)"
        )

        prompt = f"""Analyse le document suivant et produis le JSON structuré demandé.

    Exigences de résumé :
    - Le champ "summary" doit faire environ {target_words} mots
    - Longueur minimale obligatoire : {min_words} mots
        - Structure OBLIGATOIRE avec ces titres exacts :
            {'\n      '.join(SUMMARY_SECTION_TITLES)}
        - Sous chaque titre: 2 à 5 phrases factuelles (pas de liste à puces)

        Exigences de résumés par section :
        - Ajouter "section_summaries": liste d'objets {{"section_title": "...", "summary": "..."}}
        - Fournir un résumé pour chaque section détectée si possible
        - Sections détectées :
            {sections_hint}

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
        result.setdefault("section_summaries", [])
        result.setdefault("keywords", [])
        result.setdefault("classification", {"label": "Non classifié", "score": 0.0})
        result.setdefault("claims", [])

        result = await self._expand_summary_if_needed(
            parsed=parsed,
            context=context,
            result=result,
        )
        result = await self._ensure_section_summaries(
            parsed=parsed,
            context=context,
            result=result,
            section_titles=section_titles,
        )

        logger.info(
            "analyst_agent_complete",
            doc_id=parsed.doc_id,
            num_claims=len(result.get("claims", [])),
            num_keywords=len(result.get("keywords", [])),
            num_sections=len(result.get("section_summaries", [])),
            summary_words=self._summary_word_count(str(result.get("summary", ""))),
        )

        return result
