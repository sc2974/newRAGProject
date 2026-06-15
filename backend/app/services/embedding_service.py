import hashlib
import json
import math
from pathlib import Path
import re
import time

import httpx

from app.core.config import settings


TOKEN_PATTERN = re.compile(r"[\w]+|[\u4e00-\u9fff]")


class LocalEmbeddingService:
    """Small deterministic embedding for local demo indexing.

    This keeps the project runnable without downloading a model. The service can
    later be replaced by sentence-transformers or a cloud embedding provider.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OllamaEmbeddingService:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_embedding_model
        self.timeout = settings.ollama_timeout_seconds
        self._cache_dir = self._build_cache_dir()

    def embed(self, text: str) -> list[float]:
        cache_path = self._cache_path(text)
        cached_embedding = self._read_cache(cache_path)
        if cached_embedding is not None:
            return cached_embedding

        payload = {"model": self.model, "input": text}
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.base_url}/api/embed", json=payload)
                    response.raise_for_status()
                    data = response.json()
                break
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(attempt * 2)
        else:
            raise RuntimeError("Ollama embedding request failed.") from last_error

        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise ValueError("Ollama embedding response did not include embeddings.")
        embedding = [float(value) for value in embeddings[0]]
        self._write_cache(cache_path, embedding)
        return embedding

    def _build_cache_dir(self) -> Path:
        safe_model = self.model.replace(":", "_").replace("/", "_")
        cache_dir = Path(settings.document_storage_dir).parent / "embedding_cache" / safe_model
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model}\n{text}".encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> list[float] | None:
        if not path.exists():
            return None
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            return [float(value) for value in values]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _write_cache(self, path: Path, embedding: list[float]) -> None:
        try:
            path.write_text(json.dumps(embedding), encoding="utf-8")
        except OSError:
            pass
