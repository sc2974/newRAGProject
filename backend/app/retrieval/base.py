from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RetrievedChunk:
    id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float | None = None
    retrieval_method: str = "unknown"
    metadata: dict[str, float] = field(default_factory=dict)


class Retriever(Protocol):
    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        ...
