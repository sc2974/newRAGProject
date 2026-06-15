import time

from app.schemas.rag import AskRequest, AskResponse, RetrievalMode, RerankMode, SourceDocument
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.rerank_policy import (
    choose_auto_rerank_mode,
    choose_auto_retrieval_mode,
    infer_corpus_document_type,
)
from app.retrieval.reranker import KeywordOverlapReranker, SemanticCrossEncoderReranker
from app.retrieval.vector_retriever import VectorRetriever
from app.services.document_service import DocumentService
from app.services.ollama_client import OllamaClient
from app.services.reranker_warmup import get_semantic_reranker
from app.services.vector_store import VectorStore


class RagService:
    def __init__(self) -> None:
        self._vector_store = VectorStore()
        self._ollama_client = OllamaClient()
        self._document_service = DocumentService()
        self._keyword_reranker = KeywordOverlapReranker()
        self._semantic_reranker = get_semantic_reranker()
        self._last_documents = []

    async def ask(self, request: AskRequest) -> AskResponse:
        documents = await self._document_service.list_documents()
        self._last_documents = documents
        document_type = infer_corpus_document_type(documents)
        retrieval_mode = self._resolve_retrieval_mode(request, document_type)
        results, retrieval_ms = await self._search(request, retrieval_mode, documents)
        rerank_mode = self._requested_rerank_mode(request)
        max_score = self._max_score(results)
        if not results:
            return AskResponse(
                answer=(
                    "I did not find relevant document chunks yet. "
                    "Please upload and index a text document first."
                ),
                query=request.query,
                sources=[],
                used_retrieval=True,
                retrieval_status="no_results",
                max_score=max_score,
                score_threshold=request.score_threshold,
                retrieval_mode=retrieval_mode,
                rerank=rerank_mode != "none",
                rerank_mode=rerank_mode,
                retrieval_ms=retrieval_ms,
                rerank_ms=0.0,
                total_retrieval_ms=retrieval_ms,
            )

        score_threshold = self._effective_score_threshold(request, retrieval_mode)
        if score_threshold is not None:
            results = [
                result
                for result in results
                if result.score is not None and result.score >= score_threshold
            ]
        if not results:
            return AskResponse(
                answer=(
                    "I found document chunks, but their similarity scores were below "
                    "the configured threshold. I will not use them as reliable evidence."
                ),
                query=request.query,
                sources=[],
                used_retrieval=False,
                retrieval_status="below_threshold",
                max_score=max_score,
                score_threshold=score_threshold,
                retrieval_mode=retrieval_mode,
                rerank=rerank_mode != "none",
                rerank_mode=rerank_mode,
                retrieval_ms=retrieval_ms,
                rerank_ms=0.0,
                total_retrieval_ms=retrieval_ms,
            )

        rerank_mode = self._resolve_rerank_mode(request, retrieval_mode, results)

        rerank_ms = 0.0
        if rerank_mode != "none":
            rerank_started = time.perf_counter()
            reranker = self._get_reranker(rerank_mode)
            reranked = reranker.score(request.query, results[: request.candidate_limit])
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            results = [item.chunk for item in reranked[: request.limit]]
            for item, result in zip(reranked[: request.limit], results):
                result.metadata["rerank_score"] = item.score
                result.retrieval_method = f"{result.retrieval_method}+{rerank_mode}_rerank"
        else:
            results = results[: request.limit]

        sources = []
        for index, result in enumerate(results, start=1):
            citation_label = f"[{index}]"
            title = f"{result.filename} #chunk-{result.chunk_index + 1}"
            sources.append(
                SourceDocument(
                    source_id=f"{result.document_id}:chunk:{result.chunk_index}",
                    citation_label=citation_label,
                    title=title,
                    filename=result.filename,
                    content=result.text,
                    content_preview=self._excerpt(result.text, limit=320),
                    score=result.score,
                    rerank_score=result.metadata.get("rerank_score"),
                    retrieval_method=result.retrieval_method,
                    document_id=result.document_id,
                    chunk_index=result.chunk_index,
                )
            )

        prompt = self._build_prompt(request.query, sources)
        try:
            answer = await self._ollama_client.generate(prompt)
            if not answer:
                raise ValueError("Ollama returned an empty response.")
            generated_by = f"ollama:{self._ollama_client.model}"
            llm_error = None
        except Exception as exc:
            answer = self._build_answer(request.query, sources)
            generated_by = "retrieval-template"
            llm_error = str(exc)

        return AskResponse(
            answer=answer,
            query=request.query,
            sources=sources,
            used_retrieval=True,
            retrieval_status="hit",
            max_score=max_score,
            score_threshold=score_threshold,
            retrieval_mode=retrieval_mode,
            rerank=rerank_mode != "none",
            rerank_mode=rerank_mode,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            total_retrieval_ms=retrieval_ms + rerank_ms,
            generated_by=generated_by,
            llm_error=llm_error,
        )

    async def _search(
        self,
        request: AskRequest,
        retrieval_mode: RetrievalMode,
        documents,
    ):
        vector_retriever = VectorRetriever(self._vector_store)
        if retrieval_mode == "vector":
            started = time.perf_counter()
            results = vector_retriever.search(request.query, request.candidate_limit)
            return results, (time.perf_counter() - started) * 1000

        chunks = []
        for document in documents:
            document_chunks = await self._document_service.list_chunks(document.id)
            if document_chunks:
                chunks.extend(document_chunks)

        bm25_retriever = BM25Retriever(chunks)
        if retrieval_mode == "bm25":
            started = time.perf_counter()
            results = bm25_retriever.search(request.query, request.candidate_limit)
            return results, (time.perf_counter() - started) * 1000

        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
        started = time.perf_counter()
        results = hybrid_retriever.search(request.query, request.candidate_limit)
        return results, (time.perf_counter() - started) * 1000

    def _effective_score_threshold(
        self,
        request: AskRequest,
        retrieval_mode: RetrievalMode,
    ) -> float | None:
        if request.score_threshold is not None:
            return request.score_threshold
        if retrieval_mode == "vector":
            return 0.25
        return None

    def _build_prompt(self, query: str, sources: list[SourceDocument]) -> str:
        context = "\n\n".join(
            f"{source.citation_label} {source.title}\n{source.content}"
            for source in sources
        )
        return (
            "You are a retrieval-augmented QA assistant. Answer the question using ONLY "
            "the provided context. If the context does not contain the answer, say that "
            "the answer was not found in the uploaded documents. Keep the answer concise, "
            "and cite supporting sources with labels like [1] or [2].\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    def _build_answer(self, query: str, sources: list[SourceDocument]) -> str:
        top_sources = sources[:3]
        context_lines = []
        for source in top_sources:
            context_lines.append(
                f"{source.citation_label} {source.title}: {source.content_preview}"
            )

        context = "\n".join(context_lines)
        return (
            f"Based on the retrieved document chunks, the query '{query}' is most related "
            f"to the following context:\n{context}\n\n"
            "This answer is generated from retrieval results only. A real LLM generation "
            "step will be connected next."
        )

    def _excerpt(self, text: str, limit: int = 220) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}..."

    def _max_score(self, results) -> float | None:
        scores = [result.score for result in results if result.score is not None]
        if not scores:
            return None
        return max(scores)

    def _requested_rerank_mode(self, request: AskRequest) -> RerankMode:
        if request.rerank_mode == "auto":
            return "none"
        if request.rerank_mode is not None:
            return request.rerank_mode
        return "keyword" if request.rerank else "none"

    def _resolve_rerank_mode(
        self,
        request: AskRequest,
        retrieval_mode: RetrievalMode,
        results,
    ) -> RerankMode:
        if request.rerank_mode == "auto":
            return choose_auto_rerank_mode(
                retrieval_mode=retrieval_mode,
                results=results,
                document_type=infer_corpus_document_type(self._last_documents),
            )
        return self._requested_rerank_mode(request)

    def _get_reranker(self, mode: RerankMode):
        if mode == "semantic":
            return self._semantic_reranker
        return self._keyword_reranker

    def _resolve_retrieval_mode(
        self,
        request: AskRequest,
        document_type,
    ) -> RetrievalMode:
        if request.retrieval_mode == "auto":
            return choose_auto_retrieval_mode(document_type=document_type)
        return request.retrieval_mode
