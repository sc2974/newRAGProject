from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


MessageRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    id: str
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatSessionRead(BaseModel):
    id: str
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatSessionCreate(BaseModel):
    title: str = "New chat"


class ChatMessageCreate(BaseModel):
    content: str = Field(..., min_length=1)


class ChatReplyResponse(BaseModel):
    session: ChatSessionRead
    reply: ChatMessage


class ChatDeleteResponse(BaseModel):
    id: str
    deleted: bool
