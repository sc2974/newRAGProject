from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.service import AgentService
from app.schemas.agent import AgentAskRequest, AgentAskResponse


router = APIRouter()
agent_service = AgentService()


@router.post("/ask", response_model=AgentAskResponse)
async def ask(request: AgentAskRequest) -> AgentAskResponse:
    return await agent_service.ask(request)


@router.post("/ask/stream")
async def ask_stream(request: AgentAskRequest) -> StreamingResponse:
    return StreamingResponse(
        agent_service.stream(request),
        media_type="text/event-stream",
    )
