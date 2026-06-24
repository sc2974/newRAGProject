from typing import Any

from pydantic import BaseModel, Field


class AgentAskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None


class AgentToolStep(BaseModel):
    tool: str
    tool_input: Any = None
    tool_output: str
    thought: str | None = None


class AgentAskResponse(BaseModel):
    query: str
    answer: str
    generated_by: str
    steps: list[AgentToolStep] = Field(default_factory=list)
    error: str | None = None
