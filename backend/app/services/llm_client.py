import httpx

from app.core.config import settings


class LLMClient:
    def __init__(self) -> None:
        self.provider = settings.llm_provider.strip().lower()
        self.model = self._model_name()
        self.timeout = self._timeout_seconds()

    async def generate(self, prompt: str) -> str:
        if self.provider == "dashscope":
            return await self._generate_with_dashscope(prompt)
        if self.provider == "ollama":
            return await self._generate_with_ollama(prompt)
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
            "Use 'ollama' or 'dashscope'."
        )

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    async def _generate_with_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 320,
            },
            "keep_alive": "10m",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return str(data.get("response", "")).strip()

    async def _generate_with_dashscope(self, prompt: str) -> str:
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required when LLM_PROVIDER=dashscope.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{settings.dashscope_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content", "")).strip()

    def _model_name(self) -> str:
        if self.provider == "dashscope":
            return settings.dashscope_model
        return settings.ollama_model

    def _timeout_seconds(self) -> float:
        if self.provider == "dashscope":
            return settings.dashscope_timeout_seconds
        return settings.ollama_timeout_seconds
