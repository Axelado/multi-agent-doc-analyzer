"""Agent 1 — ParserAgent : ingestion et parsing documentaire."""

from __future__ import annotations

import structlog

from app.models.schemas import ParsedDocument
from app.services.document_parser import get_document_parser_service

logger = structlog.get_logger(__name__)


class ParserAgent:
    """
    Rôle : ingestion et parsing documentaire.

    Entrée : fichier brut (chemin).
    Sortie :
      - texte par page,
      - blocs structurés (titres, sections, paragraphes),
      - tables détectées,
      - métadonnées (nom, taille, nb pages, langue estimée).
    """

    def __init__(self) -> None:
        self.parser = get_document_parser_service()

    async def run(self, file_path: str, doc_id: str) -> ParsedDocument:
        """Exécuter le parsing d'un document."""
        logger.info("parser_agent_start", file_path=file_path, doc_id=doc_id)

        try:
            parsed = self.parser.parse(file_path, doc_id)

            logger.info(
                "parser_agent_complete",
                doc_id=doc_id,
                num_pages=parsed.num_pages,
                num_blocks=len(parsed.structured_blocks),
                language=parsed.language,
            )

            return parsed

        except Exception as e:
            logger.error("parser_agent_error", doc_id=doc_id, error=str(e))
            raise
