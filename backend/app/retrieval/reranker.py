import math
import re
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from app.core.config import settings
from app.retrieval.base import RetrievedChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class RerankResult:
    chunk: RetrievedChunk
    score: float


class Reranker(Protocol):
    def score(self, query: str, chunks: list[RetrievedChunk]) -> list[RerankResult]:
        ...


class KeywordOverlapReranker:
    def score(self, query: str, chunks: list[RetrievedChunk]) -> list[RerankResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return [RerankResult(chunk=chunk, score=chunk.score or 0.0) for chunk in chunks]

        results = []
        query_token_set = set(query_tokens)
        for chunk in chunks:
            chunk_tokens = tokenize(chunk.text)
            chunk_token_set = set(chunk_tokens)
            overlap = len(query_token_set & chunk_token_set)
            coverage = overlap / max(len(query_token_set), 1)
            density = overlap / max(len(chunk_token_set), 1)
            keyword_bonus = phrase_bonus(query, chunk.text)
            base_score = chunk.score or 0.0
            rerank_score = (
                0.45 * coverage
                + 0.25 * density
                + 0.20 * keyword_bonus
                + 0.10 * normalize_base_score(base_score)
            )
            results.append(RerankResult(chunk=chunk, score=rerank_score))

        return sorted(results, key=lambda item: item.score, reverse=True)


class SemanticCrossEncoderReranker:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.semantic_reranker_model
        self.max_length = max_length or settings.semantic_reranker_max_length
        self.candidate_limit = settings.semantic_reranker_candidate_limit
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load_lock = Lock()
        self.device = "cpu"

    def score(self, query: str, chunks: list[RetrievedChunk]) -> list[RerankResult]:
        if not chunks:
            return []

        self._load_model()
        rerank_candidates = chunks[: self.candidate_limit]
        tail_candidates = chunks[self.candidate_limit :]
        pairs = [[query, chunk.text] for chunk in rerank_candidates]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            if len(logits.shape) == 2 and logits.shape[1] > 1:
                scores = logits[:, 1]
            else:
                scores = logits.reshape(-1)

        scored = [
            RerankResult(chunk=chunk, score=float(score))
            for chunk, score in zip(rerank_candidates, scores.detach().cpu().tolist())
        ]
        reranked = sorted(scored, key=lambda item: item.score, reverse=True)
        reranked.extend(
            RerankResult(chunk=chunk, score=chunk.score or 0.0) for chunk in tail_candidates
        )
        return reranked

    def _load_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        with self._load_lock:
            if self._tokenizer is not None and self._model is not None:
                return

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            torch.set_num_threads(settings.semantic_reranker_torch_threads)
            self.device = self._resolve_device(torch)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def _resolve_device(self, torch) -> str:
        configured_device = settings.semantic_reranker_device.lower()
        if configured_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if configured_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return configured_device


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def phrase_bonus(query: str, text: str) -> float:
    normalized_query = " ".join(tokenize(query))
    normalized_text = " ".join(tokenize(text))
    if not normalized_query or not normalized_text:
        return 0.0
    if normalized_query in normalized_text:
        return 1.0
    query_tokens = normalized_query.split()
    matched = sum(1 for token in query_tokens if token in normalized_text)
    return matched / max(len(query_tokens), 1)


def normalize_base_score(score: float) -> float:
    if score <= 0:
        return 0.0
    return 1 / (1 + math.exp(-score))
