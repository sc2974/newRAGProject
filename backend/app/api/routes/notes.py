from fastapi import APIRouter, HTTPException

from app.schemas.notes import (
    NoteCreate,
    NoteDeleteResponse,
    NoteRead,
    NoteUpdate,
)
from app.services.note_service import NoteService

router = APIRouter()
note_service = NoteService()


@router.post("/", response_model=NoteRead)
async def create_note(request: NoteCreate) -> NoteRead:
    return await note_service.create_note(request)


@router.get("/", response_model=list[NoteRead])
async def list_notes() -> list[NoteRead]:
    return await note_service.list_notes()


@router.get("/{note_id}", response_model=NoteRead)
async def get_note(note_id: str) -> NoteRead:
    note = await note_service.get_note(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.put("/{note_id}", response_model=NoteRead)
async def update_note(note_id: str, request: NoteUpdate) -> NoteRead:
    note = await note_service.update_note(note_id, request)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.delete("/{note_id}", response_model=NoteDeleteResponse)
async def delete_note(note_id: str) -> NoteDeleteResponse:
    result = await note_service.delete_note(note_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return result
