from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str


class TextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[TextChunk]:
        normalized = text.strip()
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(index=len(chunks), text=chunk_text))

            if end >= len(normalized):
                break
            start = max(0, end - self.chunk_overlap)

        return chunks
