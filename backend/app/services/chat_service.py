from datetime import datetime, timezone

from app.schemas.chat import (
    ChatDeleteResponse,
    ChatMessage,
    ChatMessageCreate,
    ChatReplyResponse,
    ChatSessionCreate,
    ChatSessionRead,
)
from app.services.ids import new_id


class ChatService:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSessionRead] = {}

    async def create_session(self, request: ChatSessionCreate) -> ChatSessionRead:
        session = ChatSessionRead(id=new_id("chat"), title=request.title)
        self._sessions[session.id] = session
        return session

    async def list_sessions(self) -> list[ChatSessionRead]:
        return list(self._sessions.values())

    async def get_session(self, session_id: str) -> ChatSessionRead | None:
        return self._sessions.get(session_id)

    async def add_message(
        self,
        session_id: str,
        request: ChatMessageCreate,
    ) -> ChatReplyResponse | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None

        now = datetime.now(timezone.utc)
        user_message = ChatMessage(
            id=new_id("msg"),
            role="user",
            content=request.content,
            created_at=now,
        )
        reply = ChatMessage(
            id=new_id("msg"),
            role="assistant",
            content="This is a mock assistant reply. RAG generation will be connected later.",
        )

        session.messages.extend([user_message, reply])
        session.updated_at = datetime.now(timezone.utc)
        return ChatReplyResponse(session=session, reply=reply)

    async def delete_session(self, session_id: str) -> ChatDeleteResponse | None:
        if session_id not in self._sessions:
            return None

        del self._sessions[session_id]
        return ChatDeleteResponse(id=session_id, deleted=True)
