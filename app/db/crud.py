"""Opérations CRUD sur la base de données."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Document


async def create_document(
    db: AsyncSession,
    *,
    filename: str,
    file_path: str,
    file_size: int,
    file_type: str,
) -> Document:
    doc = Document(
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        file_type=file_type,
        status="queued",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)
    return doc


async def get_document(db: AsyncSession, doc_id: uuid.UUID) -> Document | None:
    return await db.get(Document, doc_id)


async def list_documents(
    db: AsyncSession, *, skip: int = 0, limit: int = 50
) -> tuple[int, list[Document]]:
    count_q = select(func.count()).select_from(Document)
    total = (await db.execute(count_q)).scalar() or 0

    q = select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(q)
    docs = list(result.scalars().all())
    return total, docs


async def update_document_status(
    db: AsyncSession,
    doc_id: uuid.UUID,
    *,
    status: str,
    failed_step: str | None = None,
    error_message: str | None = None,
) -> None:
    doc = await db.get(Document, doc_id)
    if doc:
        doc.status = status
        doc.failed_step = failed_step
        doc.error_message = error_message
        if status in {"done", "completed", "error"}:
            doc.completed_at = datetime.now(timezone.utc)
        await db.flush()


async def update_document_results(
    db: AsyncSession,
    doc_id: uuid.UUID,
    *,
    summary: str,
    keywords: list[str],
    classification: dict,
    claims: list[dict],
    confidence_global: float,
    processing_time_sec: float,
    num_pages: int | None = None,
    language: str | None = None,
    parsed_metadata: dict | None = None,
) -> None:
    doc = await db.get(Document, doc_id)
    if doc:
        doc.summary = summary
        doc.keywords = keywords
        doc.classification = classification
        doc.claims = claims
        doc.confidence_global = confidence_global
        doc.processing_time_sec = processing_time_sec
        doc.status = "done"
        doc.failed_step = None
        doc.error_message = None
        doc.completed_at = datetime.now(timezone.utc)
        if num_pages is not None:
            doc.num_pages = num_pages
        if language is not None:
            doc.language = language
        if parsed_metadata is not None:
            doc.parsed_metadata = parsed_metadata
        await db.flush()


async def get_stats(db: AsyncSession) -> dict:
    total = (await db.execute(select(func.count()).select_from(Document))).scalar() or 0

    async def count_statuses(statuses: list[str]) -> int:
        q = select(func.count()).select_from(Document).where(Document.status.in_(statuses))
        return (await db.execute(q)).scalar() or 0

    completed = await count_statuses(["done", "completed"])
    processing = await count_statuses(["parse", "index", "analyze", "verify", "edit", "processing"])
    pending = await count_statuses(["queued", "pending"])
    failed = await count_statuses(["error"])

    avg_conf_q = select(func.avg(Document.confidence_global)).where(
        Document.status.in_(["done", "completed"])
    )
    avg_conf = (await db.execute(avg_conf_q)).scalar()

    avg_time_q = select(func.avg(Document.processing_time_sec)).where(
        Document.status.in_(["done", "completed"])
    )
    avg_time = (await db.execute(avg_time_q)).scalar()

    return {
        "total_documents": total,
        "completed": completed,
        "processing": processing,
        "pending": pending,
        "failed": failed,
        "avg_confidence": round(avg_conf, 4) if avg_conf else None,
        "avg_processing_time": round(avg_time, 2) if avg_time else None,
    }
