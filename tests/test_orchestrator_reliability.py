"""Tests de fiabilité orchestrateur (retry, timeout, failed_step)."""

from __future__ import annotations

import asyncio

import pytest

from app.agents.orchestrator import AgentOrchestrator, PipelineExecutionError
from app.models.schemas import Chunk, PageContent, ParsedDocument, StructuredBlock


@pytest.fixture
def parsed_document() -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc-test",
        filename="test.txt",
        file_type="txt",
        num_pages=1,
        language="fr",
        pages=[PageContent(page_number=1, text="Contenu test")],
        structured_blocks=[
            StructuredBlock(
                block_type="paragraph",
                content="Contenu test",
                page_number=1,
            )
        ],
        metadata={},
    )


def test_parse_step_retry_then_success(parsed_document: ParsedDocument) -> None:
    """Une étape doit réussir après retry sans relancer tout le pipeline."""
    async def _run_test() -> None:
        orchestrator = AgentOrchestrator()

        orchestrator.settings.step_retry_parse = 2
        orchestrator.settings.step_backoff_base_sec = 0
        orchestrator.settings.step_backoff_max_sec = 0

        attempts = {"parse": 0}

        async def parse_run(file_path: str, doc_id: str):
            attempts["parse"] += 1
            if attempts["parse"] == 1:
                raise RuntimeError("Erreur transitoire parser")
            return parsed_document

        async def index_run(parsed: ParsedDocument):
            return [
                Chunk(
                    chunk_id="c_doc_1_0001",
                    doc_id=parsed.doc_id,
                    page_number=1,
                    text="Chunk de test",
                )
            ]

        async def analyze_run(parsed: ParsedDocument, chunks: list[Chunk]):
            return {
                "summary": "Résumé brouillon",
                "keywords": ["test"],
                "classification": {"label": "Test", "score": 0.9},
                "claims": [{"text": "Affirmation", "source_chunk_ids": [chunks[0].chunk_id]}],
            }

        async def verify_run(draft_analysis: dict, doc_id: str, chunks: list[Chunk]):
            return {
                "claims": [
                    {
                        "text": "Affirmation",
                        "status": "supported",
                        "evidence": [{"page": 1, "chunk_id": chunks[0].chunk_id, "quote": "Chunk de test", "score": 0.95}],
                        "confidence": 0.95,
                    }
                ],
                "claims_supported": [
                    {
                        "text": "Affirmation",
                        "status": "supported",
                        "evidence": [{"page": 1, "chunk_id": chunks[0].chunk_id, "quote": "Chunk de test", "score": 0.95}],
                        "confidence": 0.95,
                    }
                ],
                "claims_rejected": [],
                "summary_verified": True,
                "summary": "Résumé brouillon",
                "keywords": ["test"],
                "classification": {"label": "Test", "score": 0.9},
            }

        async def edit_run(verified: dict, parsed: ParsedDocument):
            return {
                "summary": "Résumé final",
                "keywords": ["test"],
                "classification": {"label": "Test", "score": 0.9},
                "claims": verified["claims_supported"],
                "confidence_global": 0.9,
            }

        orchestrator.parser_agent.run = parse_run
        orchestrator.index_agent.run = index_run
        orchestrator.analyst_agent.run = analyze_run
        orchestrator.verifier_agent.run = verify_run
        orchestrator.editor_agent.run = edit_run

        observed_steps: list[str] = []

        async def on_step_start(step: str) -> None:
            observed_steps.append(step)

        result = await orchestrator.run("/tmp/test.txt", "doc-test", on_step_start=on_step_start)

        assert attempts["parse"] == 2
        assert observed_steps.count("parse") == 2
        assert result["doc_id"] == "doc-test"
        assert result["summary"] == "Résumé final"

    asyncio.run(_run_test())


def test_step_timeout_sets_failed_step(parsed_document: ParsedDocument) -> None:
    """Un timeout d'étape doit remonter failed_step explicitement."""
    async def _run_test() -> None:
        orchestrator = AgentOrchestrator()

        orchestrator.settings.step_retry_parse = 1
        orchestrator.settings.step_timeout_parse_sec = 1

        async def parse_run(file_path: str, doc_id: str):
            await asyncio.sleep(2)
            return parsed_document

        orchestrator.parser_agent.run = parse_run

        with pytest.raises(PipelineExecutionError) as exc_info:
            await orchestrator.run("/tmp/test.txt", "doc-timeout")

        assert exc_info.value.failed_step == "parse"
        assert "Timeout" in str(exc_info.value)

    asyncio.run(_run_test())
