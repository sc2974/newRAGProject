from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.documents import DocumentChunkRead
from app.services.document_service import DocumentService


async def load_active_chunks(
    document_service: DocumentService,
) -> list[DocumentChunkRead]:
    chunks: list[DocumentChunkRead] = []
    documents = await document_service.list_documents()
    for document in documents:
        document_chunks = await document_service.list_chunks(document.id)
        if document_chunks:
            chunks.extend(document_chunks)
    return chunks


async def build_retrievers(document_service: DocumentService):
    chunks = await load_active_chunks(document_service)
    vector_retriever = VectorRetriever(document_service._vector_store)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    return {
        "vector": vector_retriever,
        "bm25": bm25_retriever,
        "hybrid": hybrid_retriever,
    }
