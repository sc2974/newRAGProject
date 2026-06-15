from collections import Counter
from dataclasses import dataclass
import math
import re

from app.retrieval.base import RetrievedChunk
from app.schemas.documents import DocumentChunkRead


LATIN_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class BM25Document:
    chunk: DocumentChunkRead
    term_frequency: Counter[str]
    length: int


class BM25Retriever:
    def __init__(
        self,
        chunks: list[DocumentChunkRead],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._documents = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._documents.append(
                BM25Document(
                    chunk=chunk,
                    term_frequency=Counter(tokens),
                    length=len(tokens),
                )
            )
        self._avg_doc_length = self._average_document_length()
        self._document_frequency = self._build_document_frequency()

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        query_terms = tokenize(query)
        if not query_terms or not self._documents:
            return []

        scored_chunks: list[RetrievedChunk] = []
        for document in self._documents:
            score = self._score(query_terms, document)
            if score <= 0:
                continue

            chunk = document.chunk
            scored_chunks.append(
                RetrievedChunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=score,
                    retrieval_method="bm25",
                    metadata={"bm25_score": score},
                )
            )

        return sorted(scored_chunks, key=lambda chunk: chunk.score or 0, reverse=True)[
            :limit
        ]

    def _score(self, query_terms: list[str], document: BM25Document) -> float:
        score = 0.0
        document_count = len(self._documents)
        for term in query_terms:
            frequency = document.term_frequency.get(term, 0)
            if frequency == 0:
                continue

            document_frequency = self._document_frequency.get(term, 0)
            idf = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + self._k1 * (
                1 - self._b + self._b * document.length / self._avg_doc_length
            )
            score += idf * (frequency * (self._k1 + 1)) / denominator

        return score

    def _average_document_length(self) -> float:
        if not self._documents:
            return 1.0
        average = sum(document.length for document in self._documents) / len(self._documents)
        return max(average, 1.0)

    def _build_document_frequency(self) -> Counter[str]:
        document_frequency: Counter[str] = Counter()
        for document in self._documents:
            document_frequency.update(document.term_frequency.keys())
        return document_frequency


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    latin_tokens = LATIN_TOKEN_PATTERN.findall(normalized)
    cjk_chars = CJK_PATTERN.findall(normalized)
    cjk_bigrams = [
        f"{first}{second}" for first, second in zip(cjk_chars, cjk_chars[1:])
    ]
    return latin_tokens + cjk_chars + cjk_bigrams
