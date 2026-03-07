"""Agent 2 — IndexAgent : indexation sémantique et préparation RAG."""

from __future__ import annotations

import structlog

from app.config import get_settings
from app.models.schemas import Chunk, ParsedDocument
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store_service

logger = structlog.get_logger(__name__)


class IndexAgent:
    """
    Rôle : indexation sémantique et préparation RAG.

    Entrée : blocs structurés (ParsedDocument).
    Sortie :
      - chunks normalisés,
      - embeddings,
      - index vectoriel,
      - identifiants de provenance (doc_id, page, chunk_id).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store_service()

    def _create_chunks(self, parsed: ParsedDocument) -> list[Chunk]:
        """Découper les blocs structurés en chunks normalisés."""
        chunks: list[Chunk] = []
        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap
        chunk_counter = 0

        for block in parsed.structured_blocks:
            text = block.content.strip()
            if not text:
                continue

            # Si le bloc est petit, un seul chunk
            if len(text) <= chunk_size:
                chunk_counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"c_{parsed.doc_id[:8]}_{block.page_number}_{chunk_counter:04d}",
                        doc_id=parsed.doc_id,
                        page_number=block.page_number,
                        text=text,
                        block_type=block.block_type,
                    )
                )
            else:
                # Découper en sous-chunks avec overlap
                words = text.split()
                # Calculer la taille approximative en mots
                avg_word_len = len(text) / max(len(words), 1)
                words_per_chunk = max(int(chunk_size / max(avg_word_len, 1)), 20)
                overlap_words = max(int(chunk_overlap / max(avg_word_len, 1)), 5)

                start = 0
                while start < len(words):
                    end = min(start + words_per_chunk, len(words))
                    chunk_text = " ".join(words[start:end])

                    chunk_counter += 1
                    chunks.append(
                        Chunk(
                            chunk_id=f"c_{parsed.doc_id[:8]}_{block.page_number}_{chunk_counter:04d}",
                            doc_id=parsed.doc_id,
                            page_number=block.page_number,
                            text=chunk_text,
                            block_type=block.block_type,
                        )
                    )

                    if end >= len(words):
                        break
                    start = end - overlap_words

        return chunks

    async def run(self, parsed: ParsedDocument) -> list[Chunk]:
        """Indexer un document parsé."""
        logger.info("index_agent_start", doc_id=parsed.doc_id)

        # 1. Créer les chunks
        chunks = self._create_chunks(parsed)
        logger.info("chunks_created", doc_id=parsed.doc_id, count=len(chunks))

        if not chunks:
            logger.warning("no_chunks_created", doc_id=parsed.doc_id)
            return chunks

        # 2. Calculer les embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        logger.info("embeddings_computed", doc_id=parsed.doc_id, count=len(embeddings))

        # 3. S'assurer que la collection existe
        self.vector_store.ensure_collection(self.embedding_service.dimension)

        # 4. Indexer dans Qdrant
        self.vector_store.index_chunks(chunks, embeddings)

        logger.info(
            "index_agent_complete",
            doc_id=parsed.doc_id,
            chunks_indexed=len(chunks),
        )

        return chunks
