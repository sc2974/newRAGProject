from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Demo API"
    app_env: str = "development"
    cors_origins: list[str] = ["http://localhost:5173"]
    chroma_persist_dir: str = "./storage/chroma"
    document_storage_dir: str = "./storage/documents"
    llm_provider: str = "dashscope"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:0.5b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_seconds: float = 180
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_api_key: str = ""
    dashscope_model: str = "qwen3-max"
    dashscope_timeout_seconds: float = 180
    semantic_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    semantic_reranker_max_length: int = 512
    semantic_reranker_candidate_limit: int = 8
    semantic_reranker_torch_threads: int = 4
    semantic_reranker_device: str = "auto"
    semantic_reranker_warmup: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
