from app.retrieval.base import RetrievedChunk
from app.schemas.documents import DocumentRead, DocumentType
from app.schemas.rag import RetrievalMode, RerankMode


def choose_auto_retrieval_mode(
    *,
    document_type: DocumentType | None = None,
) -> RetrievalMode:
    if document_type == "faq":
        return "hybrid"
    if document_type in {"legal", "qa", "scientific"}:
        return "bm25"
    if document_type in {"code", "tabular"}:
        return "bm25"
    return "hybrid"


def choose_auto_rerank_mode(
    *,
    retrieval_mode: RetrievalMode,
    results: list[RetrievedChunk],
    document_type: DocumentType | None = None,
) -> RerankMode:
    if not results:
        return "none"

    if retrieval_mode == "bm25":
        if document_type == "faq" and len(results) >= 5:
            return "semantic"
        return "none"

    if len(results) < 3:
        return "none"

    if document_type in {"qa"}:
        return "none"

    return "semantic"


def infer_corpus_document_type(documents: list[DocumentRead]) -> DocumentType | None:
    active = [document for document in documents if document.is_active]
    if not active:
        return None

    weights: dict[DocumentType, float] = {}
    for document in active:
        weight = max(float(document.chunk_count), 1.0)
        weights[document.document_type] = weights.get(document.document_type, 0.0) + weight

    return max(weights.items(), key=lambda item: item[1])[0]
