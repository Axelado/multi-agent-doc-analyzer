"""Routes API FastAPI."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.crud import (
    create_document,
    get_document,
    get_stats,
    list_documents,
)
from app.db.session import get_db
from app.models.schemas import (
    BatchUploadResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    StatsResponse,
    UploadResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api", tags=["API"])


def _validate_file(file: UploadFile) -> str:
    """Valider le type de fichier uploadé."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté: {suffix}. Formats acceptés: PDF, TXT",
        )
    return suffix.lstrip(".")


async def _save_file(file: UploadFile, doc_id: str) -> tuple[str, int]:
    """Sauvegarder le fichier uploadé sur le disque."""
    settings = get_settings()
    settings.ensure_dirs()

    upload_dir = Path(settings.upload_dir) / doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    content = await file.read()
    file_size = len(content)

    # Vérifier la taille
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux ({file_size} bytes). Max: {settings.max_upload_size_mb} MB",
        )

    with open(file_path, "wb") as f:
        f.write(content)

    return str(file_path), file_size


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload unitaire + lancement de l'analyse."""
    file_type = _validate_file(file)
    doc_id = str(uuid.uuid4())

    # Sauvegarder le fichier
    file_path, file_size = await _save_file(file, doc_id)

    # Créer l'entrée en base
    doc = await create_document(
        db,
        filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        file_type=file_type,
    )

    # Lancer la tâche Celery
    from app.worker.tasks import process_document_task

    process_document_task.delay(str(doc.id), file_path)

    logger.info("document_uploaded", doc_id=str(doc.id), filename=file.filename)

    return UploadResponse(
        doc_id=doc.id,
        filename=file.filename,
        status="pending",
        message="Document uploadé. Analyse en cours...",
    )


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload multiple + analyse de chaque document."""
    uploads = []

    for file in files:
        try:
            file_type = _validate_file(file)
            doc_id = str(uuid.uuid4())

            file_path, file_size = await _save_file(file, doc_id)

            doc = await create_document(
                db,
                filename=file.filename,
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
            )

            from app.worker.tasks import process_document_task

            process_document_task.delay(str(doc.id), file_path)

            uploads.append(
                UploadResponse(
                    doc_id=doc.id,
                    filename=file.filename,
                    status="pending",
                    message="Analyse en cours...",
                )
            )
        except HTTPException as e:
            uploads.append(
                UploadResponse(
                    doc_id=uuid.uuid4(),
                    filename=file.filename or "unknown",
                    status="error",
                    message=str(e.detail),
                )
            )

    return BatchUploadResponse(uploads=uploads)


@router.get("/analyses", response_model=DocumentListResponse)
async def list_analyses(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Historique des analyses."""
    total, docs = await list_documents(db, skip=skip, limit=limit)
    return DocumentListResponse(
        total=total,
        documents=[DocumentResponse.model_validate(d) for d in docs],
    )


@router.get("/analyses/{doc_id}", response_model=DocumentResponse)
async def get_analysis(
    doc_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Détail d'une analyse."""
    doc = await get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    return DocumentResponse.model_validate(doc)


@router.get("/stats", response_model=StatsResponse)
async def get_statistics(
    db: AsyncSession = Depends(get_db),
):
    """Statistiques globales."""
    stats = await get_stats(db)
    return StatsResponse(**stats)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """État des services (LLM, DB, VectorDB)."""
    from app.services.llm_service import get_llm_service
    from app.services.vector_store import get_vector_store_service

    services = {}

    # Vérifier Ollama/LLM
    try:
        llm_ok = await get_llm_service().health_check()
        services["llm"] = "ok" if llm_ok else "error"
    except Exception:
        services["llm"] = "error"

    # Vérifier Qdrant
    try:
        qdrant_ok = get_vector_store_service().health_check()
        services["vector_db"] = "ok" if qdrant_ok else "error"
    except Exception:
        services["vector_db"] = "error"

    # Vérifier Redis
    try:
        import redis

        settings = get_settings()
        r = redis.Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=2)
        r.ping()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "error"

    # Vérifier PostgreSQL
    try:
        from app.db.session import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        services["database"] = "ok"
    except Exception:
        services["database"] = "error"

    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"

    return HealthResponse(status=overall, services=services)
