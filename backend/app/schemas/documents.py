from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


DocumentStatus = Literal["uploaded", "parsed", "processing", "indexed", "failed"]
DocumentType = Literal["faq", "legal", "qa", "scientific", "code", "tabular", "general"]


class DocumentRead(BaseModel):
    id: str
    filename: str
    content_type: str | None = None
    size_bytes: int = 0
    content_md5: str = ""
    status: DocumentStatus = "uploaded"
    document_type: DocumentType = "general"
    document_type_confidence: float = 0.0
    text_length: int = 0
    chunk_count: int = 0
    text_preview: str = ""
    parse_error: str | None = None
    index_error: str | None = None
    duplicate_upload: bool = False
    duplicate_of: str | None = None
    replacement_upload: bool = False
    replaced_document_id: str | None = None
    version: int = 1
    is_active: bool = True
    deleted_at: datetime | None = None
    replaced_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentContentResponse(BaseModel):
    id: str
    filename: str
    text: str


class DocumentChunkRead(BaseModel):
    id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class DocumentSearchResult(BaseModel):
    id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float | None = None


class DocumentReindexResponse(BaseModel):
    document_count: int
    chunk_count: int
    failed_documents: list[str] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    id: str
    deleted: bool
