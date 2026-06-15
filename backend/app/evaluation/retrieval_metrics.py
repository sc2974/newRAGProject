from dataclasses import dataclass

from app.retrieval.base import RetrievedChunk


@dataclass
class RetrievalMetrics:
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    first_hit_rank: int | None
    mrr: float


def evaluate_retrieved_chunks(
    results: list[RetrievedChunk],
    answers: list[str],
) -> RetrievalMetrics:
    first_hit_rank = first_matching_rank(results, answers)
    return RetrievalMetrics(
        hit_at_1=first_hit_rank is not None and first_hit_rank <= 1,
        hit_at_3=first_hit_rank is not None and first_hit_rank <= 3,
        hit_at_5=first_hit_rank is not None and first_hit_rank <= 5,
        first_hit_rank=first_hit_rank,
        mrr=0.0 if first_hit_rank is None else 1 / first_hit_rank,
    )


def first_matching_rank(
    results: list[RetrievedChunk],
    answers: list[str],
) -> int | None:
    normalized_answers = [normalize_answer(answer) for answer in answers if answer.strip()]
    if not normalized_answers:
        return None

    for rank, result in enumerate(results, start=1):
        normalized_text = normalize_answer(result.text)
        if any(answer in normalized_text for answer in normalized_answers):
            return rank

    return None


def normalize_answer(text: str) -> str:
    return "".join(text.lower().split())
