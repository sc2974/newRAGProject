from fastapi import APIRouter

from app.schemas.llm import LLMAskRequest, LLMAskResponse
from app.services.llm_service import LLMService

router = APIRouter()
llm_service = LLMService()


@router.post("/ask", response_model=LLMAskResponse)
async def ask(request: LLMAskRequest) -> LLMAskResponse:
    return await llm_service.ask(request)
