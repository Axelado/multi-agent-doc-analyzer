"""Service NLI — vérification factuelle avec mDeBERTa."""

from __future__ import annotations

from typing import Any

import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Labels NLI standard
NLI_LABELS = ["contradiction", "neutral", "entailment"]


class NLIService:
    """Natural Language Inference pour vérification factuelle."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self.device = self._resolve_device(self.settings.nli_device)

    def _resolve_device(self, requested: str) -> str:
        requested_norm = requested.lower().strip()
        if requested_norm == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested_norm == "cuda" and not torch.cuda.is_available():
            logger.warning("nli_cuda_unavailable_fallback_cpu", requested=requested)
            return "cpu"
        if requested_norm in {"cpu", "cuda"}:
            return requested_norm
        logger.warning("nli_unknown_device_fallback_auto", requested=requested)
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

        logger.info("loading_nli_model", model=self.settings.nli_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.nli_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.settings.nli_model
        )
        try:
            model = model.to(self.device)
        except RuntimeError as exc:
            if not self._should_fallback_to_cpu(exc):
                raise
            self._switch_to_cpu("nli_load_fallback", exc)
            model = model.to(self.device)
        model.eval()
        self._model = model
        logger.info("nli_model_loaded", device=self.device)

    def _tokenize_pair_on_device(
        self,
        premise: str,
        hypothesis: str,
        device: str,
    ) -> dict:
        """Tokeniser une paire et déplacer explicitement les tenseurs."""
        encoded = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def _tokenize_batch_on_device(
        self,
        premises: list[str],
        hypotheses: list[str],
        device: str,
    ) -> dict:
        """Tokeniser un batch et déplacer explicitement les tenseurs."""
        encoded = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def _forward_pair(self, premise: str, hypothesis: str, device: str) -> list[float]:
        """Inférence NLI sur une paire pour un device donné."""
        model = self.model.to(device)
        self._model = model
        inputs = self._tokenize_pair_on_device(premise, hypothesis, device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        return [float(prob) for prob in probs]

    def _forward_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
        device: str,
    ) -> list[list[float]]:
        """Inférence NLI batch pour un device donné."""
        model = self.model.to(device)
        self._model = model
        inputs = self._tokenize_batch_on_device(premises, hypotheses, device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        return [[float(p) for p in row] for row in probs]

    def check_entailment(
        self, premise: str, hypothesis: str
    ) -> dict[str, float]:
        """
        Vérifier si la prémisse supporte l'hypothèse.

        Retourne: dict avec scores pour chaque label
        (contradiction, neutral, entailment).
        """
        self._load_model()
        try:
            probs = self._forward_pair(premise, hypothesis, self.device)
        except RuntimeError as exc:
            if not self._should_fallback_to_cpu(exc):
                raise

            self._switch_to_cpu("nli_inference_fallback", exc)
            probs = self._forward_pair(premise, hypothesis, self.device)

        return {label: round(prob, 4) for label, prob in zip(NLI_LABELS, probs)}

    def batch_check_entailment(
        self,
        premises: list[str],
        hypotheses: list[str],
    ) -> list[dict[str, float]]:
        """Vérification batch de paires prémisse/hypothèse."""
        self._load_model()
        results = []

        # Batch par paires de 8 pour ne pas saturer la mémoire
        batch_size = 8
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i : i + batch_size]
            batch_hypotheses = hypotheses[i : i + batch_size]

            try:
                probs = self._forward_batch(
                    batch_premises,
                    batch_hypotheses,
                    self.device,
                )
            except RuntimeError as exc:
                if not self._should_fallback_to_cpu(exc):
                    raise

                self._switch_to_cpu("nli_batch_inference_fallback", exc)
                probs = self._forward_batch(
                    batch_premises,
                    batch_hypotheses,
                    self.device,
                )

            for prob_row in probs:
                results.append(
                    {label: round(p, 4) for label, p in zip(NLI_LABELS, prob_row)}
                )

        return results

    def is_supported(self, scores: dict[str, float]) -> bool:
        """Déterminer si une affirmation est supportée."""
        return scores.get("entailment", 0) >= self.settings.nli_threshold

    def is_contradicted(self, scores: dict[str, float]) -> bool:
        """Déterminer si une affirmation est contredite."""
        return scores.get("contradiction", 0) >= self.settings.nli_threshold


# Singleton
_nli_service: NLIService | None = None


def get_nli_service() -> NLIService:
    global _nli_service
    if _nli_service is None:
        _nli_service = NLIService()
    return _nli_service
