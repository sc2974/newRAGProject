from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.schemas.documents import (
    DocumentChunkRead,
    DocumentContentResponse,
    DocumentDeleteResponse,
    DocumentRead,
    DocumentReindexResponse,
    DocumentSearchResult,
)
from app.services.document_classifier import classify_document
from app.services.ids import new_id
from app.services.text_splitter import TextSplitter
from app.services.vector_store import VectorStore


TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".text",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class StoredDocument:
    metadata: DocumentRead
    path: Path
    text: str


class DocumentService:
    def __init__(self) -> None:
        self._documents: dict[str, StoredDocument] = {}
        self._storage_dir = Path(settings.document_storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._storage_dir / "metadata.json"
        self._text_splitter = TextSplitter()
        self._vector_store = VectorStore()
        self._load_documents()

    async def upload(self, file: UploadFile) -> DocumentRead:
        content = await file.read()
        filename = Path(file.filename or "untitled").name
        content_md5 = self._md5(content)
        duplicate = self._find_duplicate(content_md5)
        if duplicate is not None:
            return duplicate.metadata.model_copy(
                update={
                    "duplicate_upload": True,
                    "duplicate_of": duplicate.metadata.id,
                }
            )

        document_id = new_id("doc")
        replaced = self._find_by_filename(filename)
        if replaced is not None:
            self._deactivate_document(replaced.metadata.id, replaced_by=document_id)

        extension = Path(filename).suffix.lower()
        stored_path = self._storage_dir / f"{document_id}{extension or '.bin'}"
        stored_path.write_bytes(content)

        text, parse_error = self._parse_text(filename, content)
        type_prediction = classify_document(filename, text) if text else None
        document = DocumentRead(
            id=document_id,
            filename=filename,
            content_type=file.content_type,
            size_bytes=len(content),
            content_md5=content_md5,
            status="parsed" if text else "uploaded",
            document_type=(
                type_prediction.document_type if type_prediction is not None else "general"
            ),
            document_type_confidence=(
                type_prediction.confidence if type_prediction is not None else 0.0
            ),
            text_length=len(text),
            text_preview=self._preview(text),
            parse_error=parse_error,
            replacement_upload=replaced is not None,
            replaced_document_id=replaced.metadata.id if replaced else None,
            version=(replaced.metadata.version + 1) if replaced else 1,
        )

        if text:
            try:
                chunks = self._text_splitter.split(text)
                indexed_chunks = self._vector_store.add_document_chunks(
                    document_id=document.id,
                    filename=document.filename,
                    chunks=chunks,
                )
                document.chunk_count = len(indexed_chunks)
                document.status = "indexed" if indexed_chunks else "parsed"
            except Exception as exc:  # Chroma errors should not hide upload success.
                document.status = "failed"
                document.index_error = str(exc)

        self._documents[document.id] = StoredDocument(
            metadata=document,
            path=stored_path,
            text=text,
        )
        self._persist_documents()
        return document

    async def list_documents(self) -> list[DocumentRead]:
        return [
            stored.metadata
            for stored in self._documents.values()
            if stored.metadata.is_active
        ]

    async def get_document(self, document_id: str) -> DocumentRead | None:
        stored = self._documents.get(document_id)
        if stored is None or not stored.metadata.is_active:
            return None
        return stored.metadata if stored else None

    async def get_content(self, document_id: str) -> DocumentContentResponse | None:
        stored = self._documents.get(document_id)
        if stored is None or not stored.metadata.is_active:
            return None

        return DocumentContentResponse(
            id=stored.metadata.id,
            filename=stored.metadata.filename,
            text=stored.text,
        )

    async def list_chunks(self, document_id: str) -> list[DocumentChunkRead] | None:
        stored = self._documents.get(document_id)
        if stored is None or not stored.metadata.is_active:
            return None
        return self._vector_store.list_document_chunks(document_id)

    async def search(self, query: str, limit: int) -> list[DocumentSearchResult]:
        return self._vector_store.search(query=query, limit=limit)

    async def reindex_documents(self) -> DocumentReindexResponse:
        self._vector_store.reset_collection()

        chunk_count = 0
        failed_documents: list[str] = []
        for stored in self._documents.values():
            if not stored.metadata.is_active:
                continue
            if not stored.text:
                continue

            try:
                type_prediction = classify_document(
                    stored.metadata.filename,
                    stored.text,
                )
                stored.metadata.document_type = type_prediction.document_type
                stored.metadata.document_type_confidence = type_prediction.confidence
                chunks = self._text_splitter.split(stored.text)
                indexed_chunks = self._vector_store.add_document_chunks(
                    document_id=stored.metadata.id,
                    filename=stored.metadata.filename,
                    chunks=chunks,
                )
                stored.metadata.chunk_count = len(indexed_chunks)
                stored.metadata.status = "indexed" if indexed_chunks else "parsed"
                stored.metadata.index_error = None
                chunk_count += len(indexed_chunks)
            except Exception as exc:
                stored.metadata.status = "failed"
                stored.metadata.index_error = str(exc)
                failed_documents.append(stored.metadata.id)

        self._persist_documents()
        return DocumentReindexResponse(
            document_count=len(self._documents),
            chunk_count=chunk_count,
            failed_documents=failed_documents,
        )

    async def delete_document(self, document_id: str) -> DocumentDeleteResponse | None:
        stored = self._documents.get(document_id)
        if stored is None or not stored.metadata.is_active:
            return None

        self._deactivate_document(document_id)
        self._persist_documents()
        return DocumentDeleteResponse(id=document_id, deleted=True)

    def _parse_text(self, filename: str, content: bytes) -> tuple[str, str | None]:
        extension = Path(filename).suffix.lower()
        if extension not in TEXT_EXTENSIONS:
            return "", f"Unsupported text parser for '{extension or 'unknown'}' files."

        for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
            try:
                return content.decode(encoding), None
            except UnicodeDecodeError:
                continue

        return "", "Unable to decode file as text."

    def _preview(self, text: str, limit: int = 240) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}..."

    def _md5(self, content: bytes) -> str:
        return hashlib.md5(content).hexdigest()

    def _find_duplicate(self, content_md5: str) -> StoredDocument | None:
        if not content_md5:
            return None

        for stored in self._documents.values():
            if stored.metadata.is_active and stored.metadata.content_md5 == content_md5:
                return stored

        return None

    def _find_by_filename(self, filename: str) -> StoredDocument | None:
        normalized = filename.casefold()
        for stored in self._documents.values():
            if (
                stored.metadata.is_active
                and stored.metadata.filename.casefold() == normalized
            ):
                return stored

        return None

    def _deactivate_document(
        self,
        document_id: str,
        *,
        replaced_by: str | None = None,
    ) -> StoredDocument | None:
        stored = self._documents.get(document_id)
        if stored is None:
            return None

        stored.metadata.is_active = False
        stored.metadata.deleted_at = datetime.now(timezone.utc)
        stored.metadata.replaced_by = replaced_by
        self._vector_store.delete_document(document_id)
        return stored

    def _load_documents(self) -> None:
        if not self._metadata_path.exists():
            return

        try:
            records = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        metadata_changed = False
        for record in records:
            try:
                metadata = DocumentRead.model_validate(record["metadata"])
                stored_path = self._storage_dir / record["stored_filename"]
                text = ""
                if stored_path.exists():
                    stored_bytes = stored_path.read_bytes()
                    text, _ = self._parse_text(metadata.filename, stored_bytes)
                    if not metadata.content_md5:
                        metadata.content_md5 = self._md5(stored_bytes)
                        metadata_changed = True
                    if text and (
                        metadata.document_type == "general"
                        or metadata.document_type_confidence <= 0
                    ):
                        type_prediction = classify_document(metadata.filename, text)
                        metadata.document_type = type_prediction.document_type
                        metadata.document_type_confidence = type_prediction.confidence
                        metadata_changed = True

                self._documents[metadata.id] = StoredDocument(
                    metadata=metadata,
                    path=stored_path,
                    text=text,
                )
            except (KeyError, OSError, ValueError):
                continue

        if metadata_changed:
            self._persist_documents()

    def _persist_documents(self) -> None:
        records = [
            {
                "metadata": stored.metadata.model_dump(mode="json"),
                "stored_filename": stored.path.name,
            }
            for stored in self._documents.values()
        ]
        self._metadata_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
