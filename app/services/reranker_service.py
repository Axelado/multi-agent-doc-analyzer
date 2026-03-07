"""Service de reranking — BAAI/bge-reranker-base."""

from __future__ import annotations

from typing import Any

import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import get_settings

logger = structlog.get_logger(__name__)


class RerankerService:
    """Reranking de passages avec un cross-encoder."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self.device = self._resolve_device(self.settings.reranker_device)

    def _resolve_device(self, requested: str) -> str:
        requested_norm = requested.lower().strip()
        if requested_norm == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested_norm == "cuda" and not torch.cuda.is_available():
            logger.warning("reranker_cuda_unavailable_fallback_cpu", requested=requested)
            return "cpu"
        if requested_norm in {"cpu", "cuda"}:
            return requested_norm
        logger.warning("reranker_unknown_device_fallback_auto", requested=requested)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _should_fallback_to_cpu(self, exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return (
            "expected all tensors to be on the same device" in msg
            or "cuda out of memory" in msg
            or ("out of memory" in msg and "cuda" in msg)
        )

    def _switch_to_cpu(self, event: str, error: Exception) -> None:
        if self.device == "cpu":
            return
        try:
            if self._model is not None:
                self._model = self._model.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.device = "cpu"
        logger.warning(event, error=str(error), fallback_device="cpu")

    @property
    def model(self) -> Any:
        self._load_model()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> Any:
        self._load_model()
        assert self._tokenizer is not None
        return self._tokenizer

    def _load_model(self) -> None:
        if self._model is not None:
            return

        logger.info("loading_reranker_model", model=self.settings.reranker_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.reranker_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.settings.reranker_model
        )
        try:
            model = model.to(self.device)
        except RuntimeError as exc:
            if not self._should_fallback_to_cpu(exc):
                raise
            self._switch_to_cpu("reranker_load_fallback", exc)
            model = model.to(self.device)
        model.eval()
        self._model = model
        logger.info("reranker_model_loaded", device=self.device)

    def _tokenize_on_device(
        self,
        pairs: list[list[str]],
        device: str,
    ) -> dict:
        """Tokeniser et déplacer explicitement chaque tenseur sur le bon device."""
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def _forward_scores(self, pairs: list[list[str]], device: str) -> list[float]:
        """Exécuter l'inférence de reranking sur un device donné."""
        model = self.model.to(device)
        self._model = model
        inputs = self._tokenize_on_device(pairs, device)
        with torch.no_grad():
            raw_scores = model(**inputs).logits.squeeze(-1).cpu().tolist()

        if isinstance(raw_scores, (float, int)):
            return [float(raw_scores)]
        return [float(score) for score in raw_scores]

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Reranker les passages par pertinence.

        Retourne: liste de (index_original, score) triée par score décroissant.
        """
        self._load_model()
        if not passages:
            return []

        pairs = [[query, passage] for passage in passages]

        try:
            scores = self._forward_scores(pairs, self.device)
        except RuntimeError as exc:
            if not self._should_fallback_to_cpu(exc):
                raise

            self._switch_to_cpu("reranker_inference_fallback", exc)
            scores = self._forward_scores(pairs, self.device)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores


# Singleton
_reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
