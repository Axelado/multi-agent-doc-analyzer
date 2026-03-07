"""Service de parsing documentaire — Docling + PyMuPDF fallback."""

from __future__ import annotations

from pathlib import Path

import structlog

from app.config import get_settings
from app.models.schemas import PageContent, ParsedDocument, StructuredBlock

logger = structlog.get_logger(__name__)


class DocumentParserService:
    """Extraction du contenu depuis PDF et TXT."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def parse(self, file_path: str, doc_id: str) -> ParsedDocument:
        """Parser un document (PDF ou TXT)."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(path, doc_id)
        elif suffix == ".txt":
            return self._parse_txt(path, doc_id)
        else:
            raise ValueError(f"Format non supporté: {suffix}")

    def _parse_pdf(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parser un PDF avec PyMuPDF (fallback robuste)."""
        try:
            return self._parse_pdf_pymupdf(path, doc_id)
        except Exception as e:
            logger.error("pdf_parse_error", error=str(e), path=str(path))
            raise

    def _normalize_table_data(self, table_data: list | None) -> list[list[str]]:
        """Normaliser les cellules de table pour respecter le schéma Pydantic."""
        if not table_data:
            return []

        normalized: list[list[str]] = []
        for row in table_data:
            if row is None:
                continue

            normalized_row = ["" if cell is None else str(cell).strip() for cell in row]
            normalized.append(normalized_row)

        return normalized

    def _parse_pdf_pymupdf(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parser avec PyMuPDF."""
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages = []
        structured_blocks = []
        full_text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text = page.get_text("text")
            text = raw_text if isinstance(raw_text, str) else str(raw_text or "")
            full_text += text + "\n"

            # Extraire les tables si possible
            tables = []
            try:
                find_tables = getattr(page, "find_tables", None)
                if callable(find_tables):
                    page_tables = find_tables()
                    table_iterable = (
                        page_tables.tables
                        if hasattr(page_tables, "tables")
                        else page_tables
                    )
                    for table in table_iterable:
                        table_data = table.extract()
                        normalized_table = self._normalize_table_data(table_data)
                        if normalized_table:
                            tables.append(normalized_table)
            except Exception:
                pass

            pages.append(
                PageContent(
                    page_number=page_num + 1,
                    text=text,
                    tables=tables,
                )
            )

            # Extraire les blocs structurés
            blocks_raw = page.get_text("dict")
            blocks_data = blocks_raw.get("blocks", []) if isinstance(blocks_raw, dict) else []
            for block in blocks_data:
                if block.get("type") == 0:  # Texte
                    block_text = ""
                    is_title = False
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                            # Heuristique : taille de police > 14 = titre
                            if span.get("size", 12) > 14:
                                is_title = True
                        block_text += " "

                    block_text = block_text.strip()
                    if block_text:
                        structured_blocks.append(
                            StructuredBlock(
                                block_type="title" if is_title else "paragraph",
                                content=block_text,
                                page_number=page_num + 1,
                            )
                        )

            # Ajouter les tables comme blocs
            for table_data in tables:
                table_str = "\n".join(
                    [" | ".join([cell if cell else "" for cell in row]) for row in table_data]
                )
                if table_str.strip():
                    structured_blocks.append(
                        StructuredBlock(
                            block_type="table",
                            content=table_str,
                            page_number=page_num + 1,
                        )
                    )

        # Détecter la langue
        language = self._detect_language(full_text[:2000])

        doc.close()

        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type="pdf",
            num_pages=len(pages),
            language=language,
            pages=pages,
            structured_blocks=structured_blocks,
            metadata={
                "file_size": path.stat().st_size,
                "parser": "pymupdf",
            },
        )

    def _parse_txt(self, path: Path, doc_id: str) -> ParsedDocument:
        """Parser un fichier TXT."""
        text = path.read_text(encoding="utf-8", errors="replace")

        # Découper en "pages" logiques (par blocs de ~3000 chars)
        page_size = 3000
        pages = []
        for i in range(0, len(text), page_size):
            chunk = text[i : i + page_size]
            pages.append(
                PageContent(
                    page_number=(i // page_size) + 1,
                    text=chunk,
                )
            )

        # Blocs structurés : chaque paragraphe
        structured_blocks = []
        paragraphs = text.split("\n\n")
        page_idx = 0
        char_count = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            char_count += len(para)
            page_idx = (char_count // page_size) + 1

            # Heuristique titre : ligne courte, pas de point final
            is_title = len(para) < 100 and not para.endswith(".")
            structured_blocks.append(
                StructuredBlock(
                    block_type="title" if is_title else "paragraph",
                    content=para,
                    page_number=page_idx,
                )
            )

        language = self._detect_language(text[:2000])

        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type="txt",
            num_pages=len(pages),
            language=language,
            pages=pages,
            structured_blocks=structured_blocks,
            metadata={
                "file_size": path.stat().st_size,
                "parser": "txt",
            },
        )

    def _detect_language(self, text: str) -> str | None:
        """Détecter la langue du texte."""
        try:
            from langdetect import detect

            return detect(text)
        except Exception:
            return None


# Singleton
_parser_service: DocumentParserService | None = None


def get_document_parser_service() -> DocumentParserService:
    global _parser_service
    if _parser_service is None:
        _parser_service = DocumentParserService()
    return _parser_service
