"""Worker Celery — exécution asynchrone des tâches d'analyse."""

from __future__ import annotations

import asyncio

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "nlp_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Paris",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Un seul worker pour limiter la mémoire GPU
    worker_concurrency=1,
)


_worker_loop: asyncio.AbstractEventLoop | None = None


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    """Obtenir une boucle asyncio persistante pour ce process Celery."""
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


def run_async(coro):
    """Helper pour exécuter une coroutine dans Celery."""
    loop = _get_worker_loop()
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    return loop.run_until_complete(coro)


@celery_app.task(
    bind=True,
    name="process_document",
    max_retries=2,
    default_retry_delay=30,
)
def process_document_task(self, doc_id: str, file_path: str) -> dict:
    """
    Tâche Celery : exécuter le pipeline multi-agent complet.

    Cette tâche est exécutée en arrière-plan par le worker Celery.
    """
    import structlog

    logger = structlog.get_logger(__name__)
    logger.info("celery_task_start", doc_id=doc_id, file_path=file_path)

    try:
        # Mettre à jour le statut en "processing"
        run_async(_update_status(doc_id, "processing"))

        # Exécuter le pipeline multi-agent
        from app.agents.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        result = run_async(orchestrator.run(file_path, doc_id))

        # Sauvegarder les résultats en base
        run_async(_save_results(doc_id, result))

        logger.info(
            "celery_task_complete",
            doc_id=doc_id,
            confidence=result.get("confidence_global"),
            processing_time=result.get("processing_time_sec"),
        )

        return {
            "doc_id": doc_id,
            "status": "completed",
            "confidence": result.get("confidence_global"),
        }

    except Exception as exc:
        logger.error("celery_task_error", doc_id=doc_id, error=str(exc))
        run_async(_update_status(doc_id, "error", error_message=str(exc)))

        # Retry si possible
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {"doc_id": doc_id, "status": "error", "error": str(exc)}


async def _update_status(
    doc_id: str, status: str, error_message: str | None = None
) -> None:
    """Mettre à jour le statut d'un document en base."""
    import uuid

    from app.db.crud import update_document_status
    from app.db.session import async_session_factory

    async with async_session_factory() as session:
        await update_document_status(
            session,
            uuid.UUID(doc_id),
            status=status,
            error_message=error_message,
        )
        await session.commit()


async def _save_results(doc_id: str, result: dict) -> None:
    """Sauvegarder les résultats d'analyse en base."""
    import uuid

    from app.db.crud import update_document_results
    from app.db.session import async_session_factory

    metadata = result.pop("_metadata", {})

    async with async_session_factory() as session:
        await update_document_results(
            session,
            uuid.UUID(doc_id),
            summary=result.get("summary", ""),
            keywords=result.get("keywords", []),
            classification=result.get("classification", {}),
            claims=result.get("claims", []),
            confidence_global=result.get("confidence_global", 0.0),
            processing_time_sec=result.get("processing_time_sec", 0.0),
            num_pages=metadata.get("num_pages"),
            language=metadata.get("language"),
            parsed_metadata=metadata.get("parsed_metadata"),
        )
        await session.commit()
