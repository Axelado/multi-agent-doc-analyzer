"""Agent 4 — VerifierAgent : validation factuelle anti-hallucination."""

from __future__ import annotations

import structlog

from app.config import get_settings
from app.models.schemas import Chunk, Claim, EvidenceItem
from app.services.embedding_service import get_embedding_service
from app.services.nli_service import get_nli_service
from app.services.reranker_service import get_reranker_service
from app.services.vector_store import get_vector_store_service

logger = structlog.get_logger(__name__)


class VerifierAgent:
    """
    Rôle : valider la factualité et la cohérence des affirmations.

    Méthode :
    1. Retrieval top-k sur l'index vectoriel
    2. Reranking des passages
    3. Vérification entailment / contradiction (NLI)
    4. Rejet ou correction des phrases non supportées

    Sortie :
    - claims_supported[]
    - claims_rejected[]
    - evidence_map[]
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store_service()
        self.reranker = get_reranker_service()
        self.nli = get_nli_service()

    def _find_evidence(
        self, claim_text: str, doc_id: str
    ) -> list[dict]:
        """Trouver les passages les plus pertinents pour une affirmation."""
        # 1. Retrieval sémantique
        query_vector = self.embedding_service.embed_query(claim_text)
        candidates = self.vector_store.search(
            query_vector=query_vector,
            doc_id=doc_id,
            top_k=self.settings.top_k_retrieval,
        )

        if not candidates:
            return []

        # 2. Reranking
        passages = [c["text"] for c in candidates]
        reranked = self.reranker.rerank(
            query=claim_text,
            passages=passages,
            top_k=self.settings.top_k_rerank,
        )

        # Reconstruire avec les bons indices
        evidence = []
        for orig_idx, score in reranked:
            candidate = candidates[orig_idx]
            evidence.append({
                **candidate,
                "rerank_score": score,
            })

        return evidence

    def _verify_claim(
        self, claim_text: str, evidence_passages: list[dict]
    ) -> tuple[str, list[EvidenceItem], float]:
        """Vérifier une affirmation contre les preuves."""
        if not evidence_passages:
            return "rejected", [], 0.0

        verified_evidence: list[EvidenceItem] = []
        max_entailment = 0.0
        any_contradiction = False

        for passage in evidence_passages:
            premise = passage["text"]
            nli_scores = self.nli.check_entailment(premise, claim_text)

            entailment_score = nli_scores.get("entailment", 0.0)
            contradiction_score = nli_scores.get("contradiction", 0.0)

            if self.nli.is_contradicted(nli_scores):
                any_contradiction = True
                logger.debug(
                    "claim_contradiction_detected",
                    claim=claim_text[:80],
                    passage=premise[:80],
                    scores=nli_scores,
                )

            if entailment_score > 0.3:  # Seuil souple pour les preuves
                verified_evidence.append(
                    EvidenceItem(
                        page=passage["page_number"],
                        chunk_id=passage["chunk_id"],
                        quote=premise[:300],
                        score=entailment_score,
                    )
                )
                max_entailment = max(max_entailment, entailment_score)

        # Décision finale
        if any_contradiction and max_entailment < self.settings.nli_threshold:
            status = "rejected"
            confidence = 1.0 - max_entailment
        elif max_entailment >= self.settings.nli_threshold:
            status = "supported"
            confidence = max_entailment
        else:
            status = "unverified"
            confidence = max_entailment

        return status, verified_evidence, confidence

    async def run(
        self, draft_analysis: dict, doc_id: str, chunks: list[Chunk]
    ) -> dict:
        """Vérifier toutes les affirmations du brouillon analytique."""
        logger.info("verifier_agent_start", doc_id=doc_id)

        raw_claims = draft_analysis.get("claims", [])
        verified_claims: list[Claim] = []
        claims_supported = []
        claims_rejected = []

        for raw_claim in raw_claims:
            claim_text = raw_claim.get("text", "") if isinstance(raw_claim, dict) else str(raw_claim)

            if not claim_text.strip():
                continue

            # Trouver les preuves
            evidence_passages = self._find_evidence(claim_text, doc_id)

            # Vérifier avec NLI
            status, evidence, confidence = self._verify_claim(
                claim_text, evidence_passages
            )

            claim = Claim(
                text=claim_text,
                status=status,
                evidence=evidence,
                confidence=confidence,
            )
            verified_claims.append(claim)

            if status == "supported":
                claims_supported.append(claim)
            elif status == "rejected":
                claims_rejected.append(claim)

        # Vérifier aussi le résumé (chaque phrase)
        summary = draft_analysis.get("summary", "")
        summary_sentences = [s.strip() for s in summary.split(".") if len(s.strip()) > 20]

        summary_verified = True
        for sentence in summary_sentences[:10]:  # Limiter à 10 phrases
            evidence_passages = self._find_evidence(sentence, doc_id)
            if evidence_passages:
                # Quick check
                best_passage = evidence_passages[0]
                nli_scores = self.nli.check_entailment(best_passage["text"], sentence)
                if self.nli.is_contradicted(nli_scores):
                    summary_verified = False
                    logger.warning(
                        "summary_sentence_contradicted",
                        sentence=sentence[:80],
                    )

        result = {
            "claims": [c.model_dump() for c in verified_claims],
            "claims_supported": [c.model_dump() for c in claims_supported],
            "claims_rejected": [c.model_dump() for c in claims_rejected],
            "summary_verified": summary_verified,
            "summary": draft_analysis.get("summary", ""),
            "keywords": draft_analysis.get("keywords", []),
            "classification": draft_analysis.get("classification", {}),
        }

        logger.info(
            "verifier_agent_complete",
            doc_id=doc_id,
            total_claims=len(verified_claims),
            supported=len(claims_supported),
            rejected=len(claims_rejected),
            summary_ok=summary_verified,
        )

        return result
