"""Tests de régression pour dépendances critiques et compatibilité API."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import tomllib

from qdrant_client import QdrantClient


PINNED_DEPENDENCIES = {
    "qdrant-client": "1.17.0",
    "transformers": "4.57.6",
    "torch": "2.10.0",
}


def test_pyproject_has_pinned_critical_dependencies() -> None:
    """Le pyproject doit garder les versions critiques explicitement figées."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = set(pyproject["project"]["dependencies"])

    for package, version in PINNED_DEPENDENCIES.items():
        assert f"{package}=={version}" in deps


def test_installed_versions_match_pins() -> None:
    """Les versions installées doivent correspondre au verrou attendu."""
    for package, expected_version in PINNED_DEPENDENCIES.items():
        assert metadata.version(package) == expected_version


def test_qdrant_search_api_compatibility() -> None:
    """Le client Qdrant doit exposer au moins une API de recherche supportée."""
    has_search = callable(getattr(QdrantClient, "search", None))
    has_query_points = callable(getattr(QdrantClient, "query_points", None))

    assert has_search or has_query_points
