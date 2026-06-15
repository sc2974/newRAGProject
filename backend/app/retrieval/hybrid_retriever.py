from app.retrieval.base import RetrievedChunk, Retriever


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: Retriever,
        bm25_retriever: Retriever,
        *,
        rrf_k: int = 60,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._rrf_k = rrf_k

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        vector_results = self._vector_retriever.search(query=query, limit=limit)
        bm25_results = self._bm25_retriever.search(query=query, limit=limit)
        return self.fuse(vector_results, bm25_results, limit=limit)

    def fuse(
        self,
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        combined: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}

        self._merge_results(combined, scores, vector_results, "vector")
        self._merge_results(combined, scores, bm25_results, "bm25")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        results: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            chunk = combined[chunk_id]
            chunk.score = score
            chunk.retrieval_method = "hybrid"
            chunk.metadata["hybrid_score"] = score
            results.append(chunk)

        return results

    def _merge_results(
        self,
        combined: dict[str, RetrievedChunk],
        scores: dict[str, float],
        results: list[RetrievedChunk],
        source: str,
    ) -> None:
        for rank, chunk in enumerate(results, start=1):
            if chunk.id not in combined:
                combined[chunk.id] = chunk

            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (self._rrf_k + rank)
            if chunk.score is not None:
                combined[chunk.id].metadata[f"{source}_score"] = chunk.score
            combined[chunk.id].metadata[f"{source}_rank"] = float(rank)
