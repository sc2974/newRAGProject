from datetime import datetime, timezone

from pydantic import BaseModel, Field


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = ""
    tags: list[str] = Field(default_factory=list)


class NoteUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    content: str | None = None
    tags: list[str] | None = None


class NoteRead(BaseModel):
    id: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NoteDeleteResponse(BaseModel):
    id: str
    deleted: bool
