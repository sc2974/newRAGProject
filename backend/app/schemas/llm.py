from pydantic import BaseModel, Field


class LLMAskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class LLMAskResponse(BaseModel):
    query: str
    answer: str
    generated_by: str
    llm_error: str | None = None
