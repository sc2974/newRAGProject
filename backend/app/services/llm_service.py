from app.schemas.llm import LLMAskRequest, LLMAskResponse
from app.services.llm_client import LLMClient


class LLMService:
    def __init__(self) -> None:
        self._llm_client = LLMClient()

    async def ask(self, request: LLMAskRequest) -> LLMAskResponse:
        prompt = self._build_prompt(request.query)
        try:
            answer = await self._llm_client.generate(prompt)
            if not answer:
                raise ValueError("LLM returned an empty response.")
            return LLMAskResponse(
                query=request.query,
                answer=answer,
                generated_by=self._llm_client.label,
            )
        except Exception as exc:
            return LLMAskResponse(
                query=request.query,
                answer="The LLM is unavailable right now.",
                generated_by=self._llm_client.label,
                llm_error=str(exc),
            )

    def _build_prompt(self, query: str) -> str:
        return (
            "You are a general-purpose AI assistant. This request does not include "
            "retrieved knowledge-base context, so do not claim that you searched or checked "
            "the current knowledge base. Answer from your general knowledge when possible. "
            "If you are unsure, say so directly instead of inventing details.\n\n"
            f"User question:\n{query}\n\n"
            "Answer:"
        )
