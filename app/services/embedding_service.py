"""Service d'embeddings — BAAI/bge-m3."""

from __future__ import annotations

import structlog
import torch
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Calcul d'embeddings avec sentence-transformers."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: SentenceTransformer | None = None
        self.device = self._resolve_device(self.settings.embedding_device)

    def _resolve_device(self, requested: str) -> str:
        requested_norm = requested.lower().strip()
        if requested_norm == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested_norm == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "embedding_cuda_unavailable_fallback_cpu",
                requested=requested,
            )
            return "cpu"
        if requested_norm in {"cpu", "cuda"}:
            return requested_norm
        logger.warning("embedding_unknown_device_fallback_auto", requested=requested)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _should_fallback_to_cpu(self, exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return (
            "cuda out of memory" in msg
            or ("out of memory" in msg and "cuda" in msg)
            or "expected all tensors to be on the same device" in msg
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

    def _encode(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
        def _encode_inner() -> list[list[float]]:
            if batch_size is None:
                embeddings_local = self.model.encode(
                    texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            else:
                embeddings_local = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            return embeddings_local.tolist()

        try:
            return _encode_inner()
        except RuntimeError as exc:
            if not self._should_fallback_to_cpu(exc):
                raise
            self._switch_to_cpu("embedding_inference_fallback", exc)
            return _encode_inner()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(
                "loading_embedding_model",
                model=self.settings.embedding_model,
                device=self.device,
            )
            try:
                self._model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.device,
                    trust_remote_code=True,
                )
            except RuntimeError as exc:
                if not self._should_fallback_to_cpu(exc):
                    raise
                self._switch_to_cpu("embedding_load_fallback", exc)
                self._model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.device,
                    trust_remote_code=True,
                )
            logger.info("embedding_model_loaded", device=self.device)
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Calculer les embeddings pour une liste de textes."""
        logger.debug("embedding_texts", count=len(texts))
        return self._encode(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> list[float]:
        """Calculer l'embedding d'une requête unique."""
        embeddings = self._encode([query])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        """Dimension des embeddings."""
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Dimension d'embedding indisponible")
        return int(dim)


# Singleton
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
