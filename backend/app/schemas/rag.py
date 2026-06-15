from typing import Literal

from pydantic import BaseModel, Field


RetrievalStatus = Literal["hit", "no_results", "below_threshold"]
RetrievalMode = Literal["vector", "bm25", "hybrid"]
RequestedRetrievalMode = Literal["vector", "bm25", "hybrid", "auto"]
RerankMode = Literal["none", "keyword", "semantic"]
RequestedRerankMode = Literal["none", "keyword", "semantic", "auto"]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=4, ge=1, le=10)
    candidate_limit: int = Field(default=20, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    retrieval_mode: RequestedRetrievalMode = "vector"
    rerank: bool = False
    rerank_mode: RequestedRerankMode | None = None


class SourceDocument(BaseModel):
    source_id: str
    citation_label: str
    title: str
    filename: str | None = None
    content: str
    content_preview: str
    score: float | None = None
    rerank_score: float | None = None
    retrieval_method: str = "vector"
    document_id: str | None = None
    chunk_index: int | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)
    used_retrieval: bool = False
    retrieval_status: RetrievalStatus = "no_results"
    max_score: float | None = None
    score_threshold: float | None = None
    retrieval_mode: RetrievalMode = "vector"
    rerank: bool = False
    rerank_mode: RerankMode = "none"
    retrieval_ms: float | None = None
    rerank_ms: float | None = None
    total_retrieval_ms: float | None = None
    generated_by: str = "retrieval-template"
    llm_error: str | None = None
