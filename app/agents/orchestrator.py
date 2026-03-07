"""Orchestrateur multi-agent fiabilisé (retries/timeout par étape)."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

import structlog

from app.agents.analyst_agent import AnalystAgent
from app.agents.editor_agent import EditorAgent
from app.agents.index_agent import IndexAgent
from app.agents.parser_agent import ParserAgent
from app.agents.verifier_agent import VerifierAgent
from app.config import get_settings
from app.models.schemas import Chunk, ParsedDocument

logger = structlog.get_logger(__name__)

T = TypeVar("T")
StepStatusCallback = Callable[[str], Awaitable[None]]


class PipelineExecutionError(RuntimeError):
    """Erreur enrichie avec l'étape ayant échoué."""

    def __init__(self, failed_step: str, message: str):
        super().__init__(message)
        self.failed_step = failed_step


class AgentOrchestrator:
    """Pipeline multi-agent avec retries/backoff/timeout par étape."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.parser_agent = ParserAgent()
        self.index_agent = IndexAgent()
        self.analyst_agent = AnalystAgent()
        self.verifier_agent = VerifierAgent()
        self.editor_agent = EditorAgent()

    def _get_step_timeout(self, step: str) -> int:
        mapping = {
            "parse": self.settings.step_timeout_parse_sec,
            "index": self.settings.step_timeout_index_sec,
            "analyze": self.settings.step_timeout_analyze_sec,
            "verify": self.settings.step_timeout_verify_sec,
            "edit": self.settings.step_timeout_edit_sec,
        }
        return max(1, mapping.get(step, 120))

    def _get_step_retry(self, step: str) -> int:
        mapping = {
            "parse": self.settings.step_retry_parse,
            "index": self.settings.step_retry_index,
            "analyze": self.settings.step_retry_analyze,
            "verify": self.settings.step_retry_verify,
            "edit": self.settings.step_retry_edit,
        }
        return max(1, mapping.get(step, 1))

    def _get_backoff(self, attempt: int) -> int:
        delay = self.settings.step_backoff_base_sec * (2 ** max(0, attempt - 1))
        return int(min(delay, self.settings.step_backoff_max_sec))

    async def _run_step_with_retry(
        self,
        *,
        step: str,
        doc_id: str,
        operation: Callable[[], Coroutine[Any, Any, T]],
        on_step_start: StepStatusCallback | None,
    ) -> T:
        """Exécuter une étape avec timeout et retries stricts."""
        max_attempts = self._get_step_retry(step)
        timeout_sec = self._get_step_timeout(step)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            if on_step_start is not None:
                await on_step_start(step)

            logger.info(
                "pipeline_step_start",
                doc_id=doc_id,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                timeout_sec=timeout_sec,
            )

            task: asyncio.Task[T] | None = None
            try:
                task = asyncio.create_task(operation())
                result = await asyncio.wait_for(task, timeout=timeout_sec)
                logger.info(
                    "pipeline_step_complete",
                    doc_id=doc_id,
                    step=step,
                    attempt=attempt,
                )
                return result

            except asyncio.TimeoutError as exc:
                if task is not None and not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                last_error = RuntimeError(
                    f"Timeout étape '{step}' après {timeout_sec}s"
                )

            except asyncio.CancelledError:
                if task is not None and not task.done():
                    task.cancel()
                raise

            except Exception as exc:
                last_error = exc

            if attempt < max_attempts:
                backoff_sec = self._get_backoff(attempt)
                logger.warning(
                    "pipeline_step_retry",
                    doc_id=doc_id,
                    step=step,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    backoff_sec=backoff_sec,
                    error=str(last_error),
                )
                await asyncio.sleep(backoff_sec)

        error_text = str(last_error) if last_error else "erreur inconnue"
        raise PipelineExecutionError(step, f"Erreur {step}: {error_text}")

    async def run(
        self,
        file_path: str,
        doc_id: str,
        on_step_start: StepStatusCallback | None = None,
    ) -> dict:
        """Exécuter le pipeline complet d'analyse."""
        logger.info("pipeline_start", doc_id=doc_id, file_path=file_path)
        start_time = time.time()

        try:
            parsed = await self._run_step_with_retry(
                step="parse",
                doc_id=doc_id,
                operation=lambda: self.parser_agent.run(file_path=file_path, doc_id=doc_id),
                on_step_start=on_step_start,
            )

            chunks = await self._run_step_with_retry(
                step="index",
                doc_id=doc_id,
                operation=lambda: self.index_agent.run(parsed),
                on_step_start=on_step_start,
            )

            draft = await self._run_step_with_retry(
                step="analyze",
                doc_id=doc_id,
                operation=lambda: self.analyst_agent.run(parsed, chunks),
                on_step_start=on_step_start,
            )

            verified = await self._run_step_with_retry(
                step="verify",
                doc_id=doc_id,
                operation=lambda: self.verifier_agent.run(
                    draft_analysis=draft,
                    doc_id=doc_id,
                    chunks=chunks,
                ),
                on_step_start=on_step_start,
            )

            final = await self._run_step_with_retry(
                step="edit",
                doc_id=doc_id,
                operation=lambda: self.editor_agent.run(verified, parsed),
                on_step_start=on_step_start,
            )

        except PipelineExecutionError:
            raise
        except Exception as exc:
            raise PipelineExecutionError("unknown", str(exc)) from exc

        elapsed = time.time() - start_time
        final["doc_id"] = doc_id
        final["processing_time_sec"] = round(elapsed, 2)
        final["_metadata"] = {
            "num_pages": parsed.num_pages,
            "language": parsed.language,
            "parsed_metadata": parsed.metadata,
        }

        logger.info(
            "pipeline_complete",
            doc_id=doc_id,
            elapsed=round(elapsed, 2),
            confidence=final.get("confidence_global"),
        )

        return final


# Singleton
_orchestrator: AgentOrchestrator | None = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
