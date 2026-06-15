from app.schemas.llm import LLMAskRequest, LLMAskResponse
from app.services.ollama_client import OllamaClient


class LLMService:
    def __init__(self) -> None:
        self._ollama_client = OllamaClient()

    async def ask(self, request: LLMAskRequest) -> LLMAskResponse:
        prompt = self._build_prompt(request.query)
        try:
            answer = await self._ollama_client.generate(prompt)
            if not answer:
                raise ValueError("Ollama returned an empty response.")
            return LLMAskResponse(
                query=request.query,
                answer=answer,
                generated_by=f"ollama:{self._ollama_client.model}",
            )
        except Exception as exc:
            return LLMAskResponse(
                query=request.query,
                answer="The local LLM is unavailable right now.",
                generated_by=f"ollama:{self._ollama_client.model}",
                llm_error=str(exc),
            )

    def _build_prompt(self, query: str) -> str:
        return (
            "You are a helpful local AI assistant inside a RAG knowledge-base demo. "
            "Answer the user directly and concisely. In this project context, RAG means "
            "Retrieval-Augmented Generation unless the user clearly means something else. "
            "If the question requires facts you are unsure about, say so instead of "
            "inventing details.\n\n"
            f"User question:\n{query}\n\n"
            "Answer:"
        )
