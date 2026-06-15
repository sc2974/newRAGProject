from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InvalidCollectionException

from app.core.config import settings
from app.schemas.documents import DocumentChunkRead, DocumentSearchResult
from app.services.embedding_service import OllamaEmbeddingService
from app.services.text_splitter import TextChunk


class VectorStore:
    def __init__(self) -> None:
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_service = OllamaEmbeddingService()
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = self._collection_name()
        self._collection = self._open_collection()

    def _open_collection(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.ollama_embedding_model,
            },
        )

    def add_document_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[TextChunk],
    ) -> list[DocumentChunkRead]:
        if not chunks:
            return []

        ids = [self._chunk_id(document_id, chunk.index) for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        embeddings = [self._embedding_service.embed(chunk.text) for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk.index,
            }
            for chunk in chunks
        ]

        try:
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except InvalidCollectionException:
            self._collection = self._open_collection()
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        return [
            DocumentChunkRead(
                id=chunk_id,
                document_id=document_id,
                filename=filename,
                chunk_index=chunk.index,
                text=chunk.text,
            )
            for chunk_id, chunk in zip(ids, chunks)
        ]

    def list_document_chunks(self, document_id: str) -> list[DocumentChunkRead]:
        try:
            result = self._collection.get(where={"document_id": document_id})
        except InvalidCollectionException:
            self._collection = self._open_collection()
            result = self._collection.get(where={"document_id": document_id})
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        chunks = [
            DocumentChunkRead(
                id=chunk_id,
                document_id=str(metadata["document_id"]),
                filename=str(metadata["filename"]),
                chunk_index=int(metadata["chunk_index"]),
                text=document or "",
            )
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
            if metadata
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def search(self, query: str, limit: int) -> list[DocumentSearchResult]:
        query_embedding = self._embedding_service.embed(query)
        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        except InvalidCollectionException:
            self._collection = self._open_collection()
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        results: list[DocumentSearchResult] = []
        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            if not metadata:
                continue

            results.append(
                DocumentSearchResult(
                    id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=document or "",
                    score=1 - float(distance) if distance is not None else None,
                )
            )

        return results

    def delete_document(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except InvalidCollectionException:
            self._collection = self._open_collection()
            self._collection.delete(where={"document_id": document_id})

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._collection = self._open_collection()

    def _chunk_id(self, document_id: str, chunk_index: int) -> str:
        return f"{document_id}_chunk_{chunk_index}"

    def _collection_name(self) -> str:
        safe_model = settings.ollama_embedding_model.replace(":", "_").replace("-", "_")
        return f"rag_documents_{safe_model}"
