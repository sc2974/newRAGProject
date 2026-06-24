from fastapi import APIRouter

from app.api.routes import agent, chat, documents, health, llm, notes, rag

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(notes.router, prefix="/notes", tags=["notes"])
