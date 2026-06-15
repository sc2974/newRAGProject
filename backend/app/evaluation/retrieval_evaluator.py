from dataclasses import asdict, dataclass
import time

from app.evaluation.retrieval_metrics import evaluate_retrieved_chunks
from app.retrieval.base import Retriever
from app.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class RetrievalEvalCase:
    id: str
    query: str
    answers: list[str]
    expected_document: str | None = None
    tags: list[str] | None = None


@dataclass
class RetrievalEvalRow:
    case_id: str
    query: str
    mode: str
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    first_hit_rank: int | None
    mrr: float
    latency_ms: float
    top_sources: str
    error: str | None = None


def evaluate_retrievers(
    cases: list[RetrievalEvalCase],
    retrievers: dict[str, Retriever],
    *,
    top_k: int,
    output_modes: list[str] | None = None,
) -> list[RetrievalEvalRow]:
    output_modes = output_modes or list(retrievers.keys())
    rows: list[RetrievalEvalRow] = []
    for case in cases:
        mode_results, mode_latencies, mode_errors = retrieve_once_per_mode(
            case.query,
            retrievers,
            top_k=top_k,
            required_modes=output_modes,
        )
        for mode in output_modes:
            results = mode_results[mode]
            metrics = evaluate_retrieved_chunks(results, case.answers)
            rows.append(
                build_row(
                    case,
                    mode,
                    results,
                    mode_latencies[mode],
                    metrics,
                    error=mode_errors.get(mode),
                )
            )
    return rows


def retrieve_once_per_mode(
    query: str,
    retrievers: dict[str, Retriever],
    *,
    top_k: int,
    required_modes: list[str],
) -> tuple[dict[str, list], dict[str, float]]:
    results: dict[str, list] = {}
    latencies: dict[str, float] = {}
    errors: dict[str, str | None] = {}
    needs_vector = "vector" in required_modes or "hybrid" in required_modes
    needs_bm25 = "bm25" in required_modes or "hybrid" in required_modes

    if needs_vector:
        results["vector"], latencies["vector"], errors["vector"] = timed_search(
            retrievers["vector"],
            query,
            top_k,
        )
    if needs_bm25:
        results["bm25"], latencies["bm25"], errors["bm25"] = timed_search(
            retrievers["bm25"],
            query,
            top_k,
        )
    if "hybrid" in required_modes:
        hybrid_retriever = retrievers["hybrid"]
        started_at = time.perf_counter()
        if errors.get("vector") and errors.get("bm25"):
            results["hybrid"] = []
            hybrid_error = f"vector: {errors['vector']}; bm25: {errors['bm25']}"
        elif isinstance(hybrid_retriever, HybridRetriever):
            results["hybrid"] = hybrid_retriever.fuse(
                results.get("vector", []),
                results.get("bm25", []),
                limit=top_k,
            )
            hybrid_error = errors.get("vector") or errors.get("bm25")
        else:
            try:
                results["hybrid"] = hybrid_retriever.search(query=query, limit=top_k)
                hybrid_error = None
            except Exception as exc:
                results["hybrid"] = []
                hybrid_error = str(exc)
        fusion_latency_ms = (time.perf_counter() - started_at) * 1000
        latencies["hybrid"] = (
            latencies.get("vector", 0.0)
            + latencies.get("bm25", 0.0)
            + fusion_latency_ms
        )
        errors["hybrid"] = hybrid_error

    for mode in required_modes:
        if mode not in results:
            results[mode] = []
            latencies[mode] = 0.0
            errors[mode] = f"missing retriever result for mode {mode}"

    return results, latencies, errors


def timed_search(
    retriever: Retriever,
    query: str,
    top_k: int,
) -> tuple[list, float, str | None]:
    started_at = time.perf_counter()
    try:
        results = retriever.search(query=query, limit=top_k)
        error = None
    except Exception as exc:
        results = []
        error = str(exc)
    latency_ms = (time.perf_counter() - started_at) * 1000
    return results, latency_ms, error


def build_row(
    case: RetrievalEvalCase,
    mode: str,
    results: list,
    latency_ms: float,
    metrics,
    error: str | None = None,
) -> RetrievalEvalRow:
    return RetrievalEvalRow(
        case_id=case.id,
        query=case.query,
        mode=mode,
        hit_at_1=metrics.hit_at_1,
        hit_at_3=metrics.hit_at_3,
        hit_at_5=metrics.hit_at_5,
        first_hit_rank=metrics.first_hit_rank,
        mrr=metrics.mrr,
        latency_ms=latency_ms,
        top_sources=format_top_sources(results),
        error=error,
    )


def summarize_rows(rows: list[RetrievalEvalRow]) -> dict[str, dict[str, float]]:
    modes = sorted({row.mode for row in rows})
    summary: dict[str, dict[str, float]] = {}
    for mode in modes:
        mode_rows = [row for row in rows if row.mode == mode]
        if not mode_rows:
            continue

        summary[mode] = {
            "case_count": float(len(mode_rows)),
            "hit_at_1": average(row.hit_at_1 for row in mode_rows),
            "hit_at_3": average(row.hit_at_3 for row in mode_rows),
            "hit_at_5": average(row.hit_at_5 for row in mode_rows),
            "mrr": average(row.mrr for row in mode_rows),
            "avg_latency_ms": average(row.latency_ms for row in mode_rows),
            "error_rate": average(row.error is not None for row in mode_rows),
        }
    return summary


def row_to_dict(row: RetrievalEvalRow) -> dict:
    return asdict(row)


def format_top_sources(results) -> str:
    return "|".join(
        f"{result.filename}#chunk-{result.chunk_index + 1}:{result.score}"
        for result in results
    )


def average(values) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)
