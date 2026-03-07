"""Tests — ParserAgent."""

import os
import tempfile
from pathlib import Path

import pytest

from app.agents.parser_agent import ParserAgent
from app.services.document_parser import DocumentParserService


@pytest.fixture
def sample_txt_file():
    """Créer un fichier TXT temporaire pour les tests."""
    content = """Rapport économique — T3 2025

Introduction

L'économie française a connu une croissance modérée au troisième trimestre 2025,
avec un PIB en hausse de 0.3% par rapport au trimestre précédent.

Inflation

Le taux d'inflation annuel s'est établi à 2.1% en septembre 2025,
en baisse par rapport aux 2.4% enregistrés en juin.

Emploi

Le taux de chômage est resté stable à 7.2%, avec une création nette
de 45 000 emplois dans le secteur des services.

Conclusion

Les perspectives pour le dernier trimestre restent positives, portées
par la consommation des ménages et les investissements des entreprises.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.mark.asyncio
async def test_parser_agent_txt(sample_txt_file):
    """Tester le parsing d'un fichier TXT."""
    agent = ParserAgent()
    result = await agent.run(sample_txt_file, "test-doc-001")

    assert result.doc_id == "test-doc-001"
    assert result.file_type == "txt"
    assert result.num_pages >= 1
    assert len(result.pages) >= 1
    assert len(result.structured_blocks) > 0

    # Vérifier qu'on trouve du contenu
    full_text = " ".join([p.text for p in result.pages])
    assert "PIB" in full_text or "économie" in full_text


@pytest.mark.asyncio
async def test_parser_agent_metadata(sample_txt_file):
    """Tester les métadonnées du parsing."""
    agent = ParserAgent()
    result = await agent.run(sample_txt_file, "test-doc-002")

    assert result.metadata.get("parser") == "txt"
    assert result.metadata.get("file_size", 0) > 0


def test_normalize_table_data_handles_none_cells():
    """Valider la normalisation des tables PyMuPDF avec cellules nulles."""
    parser = DocumentParserService()

    raw_table = [
        ["Colonne A", None, "Colonne C"],
        [1, None, 3.14],
        None,
    ]

    normalized = parser._normalize_table_data(raw_table)

    assert normalized == [
        ["Colonne A", "", "Colonne C"],
        ["1", "", "3.14"],
    ]
