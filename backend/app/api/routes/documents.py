from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.documents import (
    DocumentChunkRead,
    DocumentContentResponse,
    DocumentDeleteResponse,
    DocumentRead,
    DocumentReindexResponse,
    DocumentSearchRequest,
    DocumentSearchResult,
)
from app.services.document_service import DocumentService

router = APIRouter()
document_service = DocumentService()


@router.post("/upload", response_model=DocumentRead)
async def upload_document(file: UploadFile = File(...)) -> DocumentRead:
    return await document_service.upload(file)


@router.get("/", response_model=list[DocumentRead])
async def list_documents() -> list[DocumentRead]:
    return await document_service.list_documents()


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(document_id: str) -> DocumentRead:
    document = await document_service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/content", response_model=DocumentContentResponse)
async def get_document_content(document_id: str) -> DocumentContentResponse:
    content = await document_service.get_content(document_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return content


@router.get("/{document_id}/chunks", response_model=list[DocumentChunkRead])
async def list_document_chunks(document_id: str) -> list[DocumentChunkRead]:
    chunks = await document_service.list_chunks(document_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return chunks


@router.post("/search", response_model=list[DocumentSearchResult])
async def search_documents(request: DocumentSearchRequest) -> list[DocumentSearchResult]:
    return await document_service.search(query=request.query, limit=request.limit)


@router.post("/reindex", response_model=DocumentReindexResponse)
async def reindex_documents() -> DocumentReindexResponse:
    return await document_service.reindex_documents()


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str) -> DocumentDeleteResponse:
    result = await document_service.delete_document(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return result
