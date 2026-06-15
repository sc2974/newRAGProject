from app.retrieval.base import RetrievedChunk
from app.services.vector_store import VectorStore


class VectorRetriever:
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._vector_store = vector_store or VectorStore()

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        results = self._vector_store.search(query=query, limit=limit)
        return [
            RetrievedChunk(
                id=result.id,
                document_id=result.document_id,
                filename=result.filename,
                chunk_index=result.chunk_index,
                text=result.text,
                score=result.score,
                retrieval_method="vector",
                metadata={"vector_score": result.score} if result.score is not None else {},
            )
            for result in results
        ]
