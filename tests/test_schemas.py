"""Tests — Schemas Pydantic."""

import uuid

import pytest

from app.models.schemas import (
    Chunk,
    Claim,
    EvidenceItem,
    ParsedDocument,
    PageContent,
    StructuredBlock,
    UploadResponse,
    DocumentResponse,
)


def test_chunk_creation():
    chunk = Chunk(
        chunk_id="c_abc12345_1_0001",
        doc_id="test-doc",
        page_number=1,
        text="Contenu du chunk.",
    )
    assert chunk.chunk_id == "c_abc12345_1_0001"
    assert chunk.block_type == "paragraph"


def test_claim_with_evidence():
    claim = Claim(
        text="Le PIB a augmenté de 3%.",
        status="supported",
        evidence=[
            EvidenceItem(
                page=5,
                chunk_id="c_123_5_0001",
                quote="Le PIB a progressé de 3% au T3.",
                score=0.92,
            )
        ],
        confidence=0.92,
    )
    assert claim.status == "supported"
    assert len(claim.evidence) == 1
    assert claim.evidence[0].score == 0.92


def test_parsed_document():
    doc = ParsedDocument(
        doc_id="test-123",
        filename="rapport.pdf",
        file_type="pdf",
        num_pages=10,
        language="fr",
        pages=[
            PageContent(page_number=1, text="Page 1 content"),
        ],
        structured_blocks=[
            StructuredBlock(
                block_type="title",
                content="Titre du rapport",
                page_number=1,
            ),
        ],
    )
    assert doc.num_pages == 10
    assert len(doc.pages) == 1
    assert doc.structured_blocks[0].block_type == "title"


def test_upload_response():
    resp = UploadResponse(
        doc_id=uuid.uuid4(),
        filename="test.pdf",
        status="pending",
        message="Analyse en cours",
    )
    assert resp.status == "pending"
