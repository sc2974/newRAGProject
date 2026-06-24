import time
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

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
from app.services.llm_client import LLMClient
from app.services.reranker_warmup import get_semantic_reranker
from app.services.vector_store import VectorStore


class RagService:
    def __init__(self) -> None:
        self._vector_store = VectorStore()
        self._llm_client = LLMClient()
        self._document_service = DocumentService()
        self._keyword_reranker = KeywordOverlapReranker()
        self._semantic_reranker = get_semantic_reranker()
        self._last_documents = []
        self._generation_chain = self._build_generation_chain()

    async def ask(self, request: AskRequest) -> AskResponse:
        documents = await self._document_service.list_documents()
        self._last_documents = documents
        document_type = infer_corpus_document_type(documents)
        retrieval_mode = self._resolve_retrieval_mode(request, document_type)
        hyde_query, hyde_ms, hyde_error = await self._generate_hyde_query(request)
        results, retrieval_ms = await self._search(request, retrieval_mode, documents)
        if hyde_query:
            hyde_results, hyde_retrieval_ms = await self._search_text(
                query=hyde_query,
                retrieval_mode=retrieval_mode,
                documents=documents,
                limit=request.candidate_limit,
            )
            retrieval_ms += hyde_retrieval_ms
            results = self._merge_hyde_results(results, hyde_results)
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
                hyde_ms=hyde_ms,
                rerank_ms=0.0,
                total_retrieval_ms=retrieval_ms,
                hyde=request.hyde,
                hyde_query=hyde_query,
                hyde_error=hyde_error,
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
                hyde_ms=hyde_ms,
                rerank_ms=0.0,
                total_retrieval_ms=retrieval_ms,
                hyde=request.hyde,
                hyde_query=hyde_query,
                hyde_error=hyde_error,
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

        try:
            answer = await self._generation_chain.ainvoke(
                {
                    "query": request.query,
                    "context": self._build_context(sources),
                    "answer_focus": self._format_answer_focus(request.query),
                }
            )
            if not answer:
                raise ValueError("LLM returned an empty response.")
            generated_by = self._llm_client.label
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
            hyde_ms=hyde_ms,
            rerank_ms=rerank_ms,
            total_retrieval_ms=retrieval_ms + hyde_ms + rerank_ms,
            generated_by=generated_by,
            llm_error=llm_error,
            hyde=request.hyde,
            hyde_query=hyde_query,
            hyde_error=hyde_error,
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

    async def _search_text(
        self,
        *,
        query: str,
        retrieval_mode: RetrievalMode,
        documents,
        limit: int,
    ):
        request = AskRequest(
            query=query,
            retrieval_mode=retrieval_mode,
            candidate_limit=limit,
            limit=min(limit, 10),
            hyde=False,
        )
        return await self._search(request, retrieval_mode, documents)

    async def _generate_hyde_query(self, request: AskRequest) -> tuple[str | None, float, str | None]:
        if not request.hyde:
            return None, 0.0, None

        started = time.perf_counter()
        try:
            hyde_query = await self._llm_client.generate(self._build_hyde_prompt(request.query))
            hyde_query = self._clean_hyde_query(hyde_query)
            if not hyde_query:
                raise ValueError("LLM returned an empty HyDE hypothesis.")
            return hyde_query, (time.perf_counter() - started) * 1000, None
        except Exception as exc:
            return None, (time.perf_counter() - started) * 1000, str(exc)

    def _build_hyde_prompt(self, query: str) -> str:
        return (
            "You are generating a HyDE retrieval hypothesis, not the final answer.\n"
            "Write a short, factual-looking passage that could plausibly appear in a source "
            "document answering the user's question. The passage is only used for semantic "
            "retrieval, so include likely entities, dates, aliases, technical terms, and "
            "relationships. Do not add citations. Do not say you are uncertain. Keep it under "
            "120 words. Use the same language as the question when possible.\n\n"
            f"Question: {query}\n\n"
            "Hypothetical source passage:"
        )

    def _clean_hyde_query(self, text: str) -> str:
        normalized = " ".join(text.strip().split())
        prefixes = (
            "Hypothetical source passage:",
            "HyDE:",
            "假设答案：",
            "假设文档：",
        )
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        return normalized[:1000]

    def _merge_hyde_results(self, original_results, hyde_results):
        combined = {}
        for rank, result in enumerate(original_results, start=1):
            result.metadata["original_rank"] = float(rank)
            result.metadata["original_score"] = result.score or 0.0
            combined[result.id] = result

        for rank, hyde_result in enumerate(hyde_results, start=1):
            existing = combined.get(hyde_result.id)
            hyde_result.metadata["hyde_rank"] = float(rank)
            hyde_result.metadata["hyde_score"] = hyde_result.score or 0.0
            if existing is None:
                hyde_result.retrieval_method = f"{hyde_result.retrieval_method}+hyde"
                combined[hyde_result.id] = hyde_result
                continue

            existing.metadata["hyde_rank"] = float(rank)
            existing.metadata["hyde_score"] = hyde_result.score or 0.0
            existing.retrieval_method = f"{existing.retrieval_method}+hyde"
            if existing.score is not None and hyde_result.score is not None:
                existing.score = max(existing.score, hyde_result.score)

        return sorted(
            combined.values(),
            key=lambda result: (
                result.score if result.score is not None else 0.0,
                -result.metadata.get("hyde_rank", 9999.0),
            ),
            reverse=True,
        )

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

    def _build_generation_chain(self):
        prompt_template = PromptTemplate.from_template(self._load_rag_prompt_template())
        return prompt_template | RunnableLambda(self._generate_from_prompt) | StrOutputParser()

    def _load_rag_prompt_template(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "rag_answer.txt"
        try:
            return prompt_path.read_text(encoding="utf-8")
        except OSError:
            return (
                "You are a retrieval-augmented QA assistant. Answer the question based on the "
                "provided context, and do not invent facts that are not supported by the "
                "context. You may paraphrase and combine information from multiple context "
                "chunks instead of copying the original wording. The final answer must directly "
                "answer the user's question and include every required answer focus when the "
                "context supports it. If the context does not contain enough evidence, say that "
                "the answer was not found in the uploaded documents. Keep the answer concise, "
                "answer in the same language as the question, and cite supporting sources with "
                "labels like [1] or [2].\n\n"
                "Required answer focus: {answer_focus}.\n"
                "Question:\n{query}\n\n"
                "Context:\n{context}\n\n"
                "Answer:"
            )

    async def _generate_from_prompt(self, prompt_value) -> str:
        return await self._llm_client.generate(prompt_value.to_string())

    def _build_context(self, sources: list[SourceDocument]) -> str:
        return "\n\n".join(
            f"{source.citation_label} {source.title}\n{source.content}"
            for source in sources
        )

    def _format_answer_focus(self, query: str) -> str:
        answer_focus = self._answer_focus(query)
        if not answer_focus:
            return "directly answer the user's question"
        return ", ".join(answer_focus)

    def _answer_focus(self, query: str) -> list[str]:
        lowered = query.lower()
        focus_patterns = [
            ("year", ("\u54ea\u4e00\u5e74", "\u51e0\u5e74", "\u4f55\u5e74", "\u5e74\u4efd", "\u5e74\u5ea6", "year")),
            ("time", ("\u4ec0\u4e48\u65f6\u5019", "\u4f55\u65f6", "\u65f6\u95f4", "when")),
            ("method", ("\u4ec0\u4e48\u65b9\u5f0f", "\u54ea\u79cd\u65b9\u5f0f", "\u4f55\u79cd\u65b9\u5f0f", "\u4ee5\u4f55\u65b9\u5f0f", "\u65b9\u5f0f", "method", "how")),
            ("place", ("\u54ea\u91cc", "\u5728\u54ea", "\u5730\u70b9", "\u4f4d\u7f6e", "\u5730\u65b9", "where", "place")),
            ("person", ("\u8c01", "\u54ea\u4f4d", "\u4f55\u4eba", "who")),
            ("reason", ("\u4e3a\u4ec0\u4e48", "\u4e3a\u4f55", "\u539f\u56e0", "reason", "why")),
            ("number", ("\u591a\u5c11", "\u51e0\u4e2a", "\u51e0\u79cd", "\u6570\u91cf", "how many", "number")),
            ("result", ("\u7ed3\u679c", "\u53d8\u5316", "\u5f71\u54cd", "\u4f5c\u7528", "result", "outcome")),
        ]

        focus = []
        for label, patterns in focus_patterns:
            if any(pattern in lowered for pattern in patterns):
                focus.append(label)

        return focus

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

