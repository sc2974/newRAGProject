from fastapi import APIRouter

from app.rag.service import RagService
from app.schemas.rag import AskRequest, AskResponse

router = APIRouter()
rag_service = RagService()


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    return await rag_service.ask(request)
