from datetime import datetime, timezone

from app.schemas.notes import (
    NoteCreate,
    NoteDeleteResponse,
    NoteRead,
    NoteUpdate,
)
from app.services.ids import new_id


class NoteService:
    def __init__(self) -> None:
        self._notes: dict[str, NoteRead] = {}

    async def create_note(self, request: NoteCreate) -> NoteRead:
        note = NoteRead(
            id=new_id("note"),
            title=request.title,
            content=request.content,
            tags=request.tags,
        )
        self._notes[note.id] = note
        return note

    async def list_notes(self) -> list[NoteRead]:
        return list(self._notes.values())

    async def get_note(self, note_id: str) -> NoteRead | None:
        return self._notes.get(note_id)

    async def update_note(self, note_id: str, request: NoteUpdate) -> NoteRead | None:
        note = self._notes.get(note_id)
        if note is None:
            return None

        update_data = request.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(note, field, value)

        note.updated_at = datetime.now(timezone.utc)
        return note

    async def delete_note(self, note_id: str) -> NoteDeleteResponse | None:
        if note_id not in self._notes:
            return None

        del self._notes[note_id]
        return NoteDeleteResponse(id=note_id, deleted=True)
