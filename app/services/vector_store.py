"""Service de base vectorielle — Qdrant."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import get_settings
from app.models.schemas import Chunk

logger = structlog.get_logger(__name__)


class VectorStoreService:
    """Gestion de l'index vectoriel Qdrant."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )
        return self._client

    def ensure_collection(self, dimension: int) -> None:
        """Créer la collection si elle n'existe pas."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.settings.qdrant_collection not in collections:
            self.client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "collection_created",
                name=self.settings.qdrant_collection,
                dimension=dimension,
            )

    def index_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Indexer les chunks avec leurs embeddings."""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "block_type": chunk.block_type,
                    },
                )
            )

        # Indexer par batch de 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=batch,
            )

        logger.info("chunks_indexed", count=len(points))

    def search(
        self,
        query_vector: list[float],
        doc_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Recherche par similarité limitée à un document."""
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        )

        search_fn = getattr(self.client, "search", None)
        query_points_fn = getattr(self.client, "query_points", None)

        if callable(search_fn):
            response = search_fn(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )
            if isinstance(response, list):
                results: list[Any] = response
            elif isinstance(response, tuple):
                results = list(response)
            else:
                try:
                    results = list(response)  # type: ignore[arg-type]
                except TypeError:
                    results = []
        elif callable(query_points_fn):
            response = query_points_fn(
                collection_name=self.settings.qdrant_collection,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )
            points = getattr(response, "points", response)
            if isinstance(points, list):
                results = points
            elif isinstance(points, tuple):
                results = list(points)
            else:
                try:
                    results = list(points)  # type: ignore[arg-type]
                except TypeError:
                    results = []
        else:
            raise RuntimeError(
                "Aucune méthode de recherche compatible trouvée sur QdrantClient"
            )

        formatted_results = []
        for hit in results:
            payload = getattr(hit, "payload", None) or {}
            score = float(getattr(hit, "score", 0.0) or 0.0)
            formatted_results.append(
                {
                    "chunk_id": payload.get("chunk_id", ""),
                    "doc_id": payload.get("doc_id", ""),
                    "page_number": payload.get("page_number", 0),
                    "text": payload.get("text", ""),
                    "block_type": payload.get("block_type", ""),
                    "score": score,
                }
            )

        return formatted_results

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Supprimer tous les vecteurs d'un document."""
        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
        logger.info("vectors_deleted", doc_id=doc_id)

    def health_check(self) -> bool:
        """Vérifier que Qdrant est accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


# Singleton
_vector_store_service: VectorStoreService | None = None


def get_vector_store_service() -> VectorStoreService:
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service
