"""Tests — AnalystAgent (longueur de résumé)."""

from __future__ import annotations

import asyncio
import json

from app.agents import analyst_agent as analyst_module
from app.models.schemas import Chunk, PageContent, ParsedDocument, StructuredBlock


class _FakeLLM:
    def __init__(
        self,
        structured_response: str,
        generated_responses: list[str] | None = None,
    ) -> None:
        self._structured_response = structured_response
        self._generated_responses = generated_responses or []
        self.structured_calls = 0
        self.generate_calls = 0

    async def generate_structured(self, prompt: str, system_prompt: str = "") -> str:
        self.structured_calls += 1
        return self._structured_response

    async def generate(self, prompt: str) -> str:
        self.generate_calls += 1
        if self._generated_responses:
            return self._generated_responses.pop(0)
        return ""


def _make_parsed_document(with_titles: bool = False) -> ParsedDocument:
    blocks = [
        StructuredBlock(
            block_type="paragraph",
            content="Contenu test",
            page_number=1,
        )
    ]
    if with_titles:
        blocks = [
            StructuredBlock(block_type="title", content="Introduction", page_number=1),
            StructuredBlock(block_type="paragraph", content="Contexte économique initial", page_number=1),
            StructuredBlock(block_type="title", content="Analyse sectorielle", page_number=2),
            StructuredBlock(block_type="paragraph", content="Détails par secteur", page_number=2),
        ]

    return ParsedDocument(
        doc_id="doc-analyst-test",
        filename="rapport.txt",
        file_type="txt",
        num_pages=2 if with_titles else 1,
        language="fr",
        pages=[PageContent(page_number=1, text="Contenu test")],
        structured_blocks=blocks,
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


def _build_structured_summary(word_count_per_section: int = 18) -> str:
    sections: list[str] = []
    token_id = 0
    for title in analyst_module.SUMMARY_SECTION_TITLES:
        words = " ".join(
            f"mot{token_id + offset}" for offset in range(word_count_per_section)
        )
        token_id += word_count_per_section
        sections.append(f"{title}\n{words}.")
    return "\n\n".join(sections)


def test_analyst_keeps_long_summary_without_expansion(monkeypatch) -> None:
    long_summary = _build_structured_summary(word_count_per_section=16)
    structured_payload = json.dumps(
        {
            "summary": long_summary,
            "section_summaries": [
                {"section_title": "Partie 1", "summary": "Résumé section 1"},
                {"section_title": "Partie 2", "summary": "Résumé section 2"},
            ],
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
    assert all(title in result["summary"] for title in analyst_module.SUMMARY_SECTION_TITLES)


def test_analyst_expands_too_short_summary(monkeypatch) -> None:
    short_summary = "Résumé beaucoup trop court pour être utile."
    structured_payload = json.dumps(
        {
            "summary": short_summary,
            "section_summaries": [
                {"section_title": "Partie 1", "summary": "Résumé section 1"},
            ],
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
        generated_responses=[expanded_summary],
    )
    monkeypatch.setattr(analyst_module, "get_llm_service", lambda: fake_llm)

    agent = analyst_module.AnalystAgent()
    agent.settings.analyst_summary_min_words = 80
    agent.settings.analyst_summary_retry_count = 1

    result = asyncio.run(agent.run(_make_parsed_document(), _make_chunks()))

    assert len(result["summary"].split()) >= 80
    assert fake_llm.generate_calls == 1


def test_analyst_reformats_unstructured_summary(monkeypatch) -> None:
    long_unstructured_summary = " ".join([f"detail{i}" for i in range(140)])
    structured_payload = json.dumps(
        {
            "summary": long_unstructured_summary,
            "section_summaries": [
                {"section_title": "Partie A", "summary": "Résumé A"},
                {"section_title": "Partie B", "summary": "Résumé B"},
            ],
            "keywords": ["croissance"],
            "classification": {"label": "Macro", "score": 0.79},
            "claims": [
                {
                    "text": "Le rapport évoque une croissance modérée.",
                    "source_chunk_ids": ["c_doc_1_0001"],
                }
            ],
        }
    )
    structured_rewrite = _build_structured_summary(word_count_per_section=20)

    fake_llm = _FakeLLM(
        structured_response=structured_payload,
        generated_responses=[structured_rewrite],
    )
    monkeypatch.setattr(analyst_module, "get_llm_service", lambda: fake_llm)

    agent = analyst_module.AnalystAgent()
    agent.settings.analyst_summary_min_words = 80
    agent.settings.analyst_summary_retry_count = 1

    result = asyncio.run(agent.run(_make_parsed_document(), _make_chunks()))

    assert fake_llm.generate_calls == 1
    assert all(title in result["summary"] for title in analyst_module.SUMMARY_SECTION_TITLES)


def test_analyst_generates_section_summaries_for_detected_parts(monkeypatch) -> None:
    long_summary = _build_structured_summary(word_count_per_section=18)
    structured_payload = json.dumps(
        {
            "summary": long_summary,
            "keywords": ["macro"],
            "classification": {"label": "Macro", "score": 0.9},
            "claims": [
                {
                    "text": "Le document couvre plusieurs volets.",
                    "source_chunk_ids": ["c_doc_1_0001"],
                }
            ],
        }
    )
    generated_sections = json.dumps(
        [
            {"section_title": "Introduction", "summary": "Résumé de l'introduction."},
            {"section_title": "Analyse sectorielle", "summary": "Résumé de l'analyse sectorielle."},
        ],
        ensure_ascii=False,
    )

    fake_llm = _FakeLLM(
        structured_response=structured_payload,
        generated_responses=[generated_sections],
    )
    monkeypatch.setattr(analyst_module, "get_llm_service", lambda: fake_llm)

    agent = analyst_module.AnalystAgent()
    agent.settings.analyst_summary_min_words = 80
    agent.settings.analyst_summary_retry_count = 1

    result = asyncio.run(agent.run(_make_parsed_document(with_titles=True), _make_chunks()))

    assert fake_llm.generate_calls == 1
    assert len(result.get("section_summaries", [])) >= 2
    produced_titles = {s.get("section_title") for s in result["section_summaries"]}
    assert "Introduction" in produced_titles
    assert "Analyse sectorielle" in produced_titles
