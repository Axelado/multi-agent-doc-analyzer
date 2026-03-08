"""Tests — AnalystAgent (longueur de résumé)."""

from __future__ import annotations

import asyncio
import json

from app.agents import analyst_agent as analyst_module
from app.models.schemas import Chunk, PageContent, ParsedDocument, StructuredBlock


class _FakeLLM:
    def __init__(self, structured_response: str, expanded_summary: str = "") -> None:
        self._structured_response = structured_response
        self._expanded_summary = expanded_summary
        self.structured_calls = 0
        self.generate_calls = 0

    async def generate_structured(self, prompt: str, system_prompt: str = "") -> str:
        self.structured_calls += 1
        return self._structured_response

    async def generate(self, prompt: str) -> str:
        self.generate_calls += 1
        return self._expanded_summary


def _make_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc-analyst-test",
        filename="rapport.txt",
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


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c_doc_1_0001",
            doc_id="doc-analyst-test",
            page_number=1,
            text="Texte de référence pour le résumé.",
        )
    ]


def test_analyst_keeps_long_summary_without_expansion(monkeypatch) -> None:
    long_summary = " ".join([f"mot{i}" for i in range(90)])
    structured_payload = json.dumps(
        {
            "summary": long_summary,
            "keywords": ["finance", "croissance"],
            "classification": {"label": "Macro", "score": 0.88},
            "claims": [
                {
                    "text": "Le rapport mentionne une progression.",
                    "source_chunk_ids": ["c_doc_1_0001"],
                }
            ],
        }
    )

    fake_llm = _FakeLLM(structured_response=structured_payload)
    monkeypatch.setattr(analyst_module, "get_llm_service", lambda: fake_llm)

    agent = analyst_module.AnalystAgent()
    agent.settings.analyst_summary_min_words = 60
    agent.settings.analyst_summary_retry_count = 1

    result = asyncio.run(agent.run(_make_parsed_document(), _make_chunks()))

    assert len(result["summary"].split()) >= 60
    assert fake_llm.generate_calls == 0


def test_analyst_expands_too_short_summary(monkeypatch) -> None:
    short_summary = "Résumé beaucoup trop court pour être utile."
    structured_payload = json.dumps(
        {
            "summary": short_summary,
            "keywords": ["inflation"],
            "classification": {"label": "Macro", "score": 0.81},
            "claims": [
                {
                    "text": "La pression inflationniste est évoquée.",
                    "source_chunk_ids": ["c_doc_1_0001"],
                }
            ],
        }
    )
    expanded_summary = " ".join([f"detail{i}" for i in range(120)])

    fake_llm = _FakeLLM(
        structured_response=structured_payload,
        expanded_summary=expanded_summary,
    )
    monkeypatch.setattr(analyst_module, "get_llm_service", lambda: fake_llm)

    agent = analyst_module.AnalystAgent()
    agent.settings.analyst_summary_min_words = 80
    agent.settings.analyst_summary_retry_count = 1

    result = asyncio.run(agent.run(_make_parsed_document(), _make_chunks()))

    assert len(result["summary"].split()) >= 80
    assert fake_llm.generate_calls == 1
