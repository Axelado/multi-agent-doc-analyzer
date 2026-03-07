"""Tests — IndexAgent (chunking)."""

import pytest

from app.agents.index_agent import IndexAgent
from app.models.schemas import ParsedDocument, PageContent, StructuredBlock


def test_chunk_creation():
    """Tester la création de chunks à partir de blocs structurés."""
    agent = IndexAgent()

    parsed = ParsedDocument(
        doc_id="test-chunk-001",
        filename="test.txt",
        file_type="txt",
        num_pages=1,
        pages=[PageContent(page_number=1, text="test")],
        structured_blocks=[
            StructuredBlock(
                block_type="paragraph",
                content="Ceci est un paragraphe court pour tester le chunking.",
                page_number=1,
            ),
            StructuredBlock(
                block_type="title",
                content="Titre de section",
                page_number=1,
            ),
            StructuredBlock(
                block_type="paragraph",
                content=" ".join(["mot"] * 200),  # Long paragraph
                page_number=2,
            ),
        ],
    )

    chunks = agent._create_chunks(parsed)

    assert len(chunks) >= 3  # At least one per block, plus overflow
    assert all(c.doc_id == "test-chunk-001" for c in chunks)
    assert all(c.chunk_id.startswith("c_") for c in chunks)
    assert any(c.block_type == "title" for c in chunks)
