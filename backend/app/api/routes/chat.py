from fastapi import APIRouter, HTTPException

from app.schemas.chat import (
    ChatDeleteResponse,
    ChatMessageCreate,
    ChatReplyResponse,
    ChatSessionCreate,
    ChatSessionRead,
)
from app.services.chat_service import chat_service

router = APIRouter()


@router.post("/sessions", response_model=ChatSessionRead)
async def create_session(request: ChatSessionCreate) -> ChatSessionRead:
    return await chat_service.create_session(request)


@router.get("/sessions", response_model=list[ChatSessionRead])
async def list_sessions() -> list[ChatSessionRead]:
    return await chat_service.list_sessions()


@router.get("/sessions/{session_id}", response_model=ChatSessionRead)
async def get_session(session_id: str) -> ChatSessionRead:
    session = await chat_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


@router.post("/sessions/{session_id}/messages", response_model=ChatReplyResponse)
async def add_message(
    session_id: str,
    request: ChatMessageCreate,
) -> ChatReplyResponse:
    result = await chat_service.add_message(session_id, request)
    if result is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return result


@router.delete("/sessions/{session_id}", response_model=ChatDeleteResponse)
async def delete_session(session_id: str) -> ChatDeleteResponse:
    result = await chat_service.delete_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return result
