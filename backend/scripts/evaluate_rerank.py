import argparse
import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
import shutil
import sys
import time

from starlette.datastructures import UploadFile


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.evaluation.retrieval_evaluator import RetrievalEvalCase, retrieve_once_per_mode
from app.evaluation.retrieval_metrics import evaluate_retrieved_chunks
from app.core.config import settings
from app.retrieval.factory import build_retrievers
from app.retrieval.rerank_policy import choose_auto_rerank_mode, infer_corpus_document_type
from app.retrieval.rerank_policy import choose_auto_retrieval_mode
from app.retrieval.reranker import KeywordOverlapReranker, SemanticCrossEncoderReranker
from app.services.document_service import DocumentService


DEFAULT_DATASET = BACKEND_DIR / "eval_data" / "faq" / "customer_support_faq_rewrite_eval.jsonl"
DEFAULT_REPORT_DIR = BACKEND_DIR / "reports" / "rerank_comparison"


@dataclass
class RerankEvalRow:
    case_id: str
    query: str
    mode: str
    rerank_mode: str
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    first_hit_rank: int | None
    mrr: float
    retrieval_ms: float
    rerank_ms: float
    total_ms: float
    candidate_count: int
    top_sources: str
    error: str | None = None


async def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(dataset_path)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    if args.corpus:
        configure_corpus_storage(args, report_dir)
    if args.document_storage_dir:
        settings.document_storage_dir = args.document_storage_dir
    if args.chroma_persist_dir:
        settings.chroma_persist_dir = args.chroma_persist_dir

    document_service = DocumentService()
    if args.corpus:
        await ensure_corpus_indexed(document_service, Path(args.corpus))
    documents = await document_service.list_documents()
    document_type = infer_corpus_document_type(documents)
    retrievers = await build_retrievers(document_service)
    rerankers = build_rerankers(args.rerank_modes)
    if args.warmup_reranker and "semantic" in rerankers:
        warmup_semantic_reranker(rerankers["semantic"])

    rows: list[RerankEvalRow] = []
    required_modes = sorted(
        {resolve_retrieval_mode(mode, document_type) for mode in args.modes}
    )
    for case in cases:
        mode_results, mode_latencies, mode_errors = retrieve_once_per_mode(
            case.query,
            retrievers,
            top_k=args.candidate_k,
            required_modes=required_modes,
        )
        for requested_mode in args.modes:
            mode = resolve_retrieval_mode(requested_mode, document_type)
            candidates = mode_results[mode]
            retrieval_ms = mode_latencies[mode]
            retrieval_error = mode_errors.get(mode)

            for rerank_mode in args.rerank_modes:
                effective_rerank_mode = resolve_rerank_mode(
                    rerank_mode,
                    mode,
                    candidates,
                    document_type,
                )
                if effective_rerank_mode == "none":
                    rows.append(
                        build_row(
                            case=case,
                            mode=requested_mode,
                            rerank_mode=rerank_mode,
                            results=candidates[: args.top_k],
                            retrieval_ms=retrieval_ms,
                            rerank_ms=0.0,
                            candidate_count=len(candidates),
                            error=retrieval_error,
                        )
                    )
                    continue

                rerank_started_at = time.perf_counter()
                reranked = rerankers[effective_rerank_mode].score(case.query, candidates)
                rerank_ms = (time.perf_counter() - rerank_started_at) * 1000
                reranked_results = [item.chunk for item in reranked[: args.top_k]]
                rows.append(
                    build_row(
                        case=case,
                        mode=requested_mode,
                        rerank_mode=rerank_mode,
                        results=reranked_results,
                        retrieval_ms=retrieval_ms,
                        rerank_ms=rerank_ms,
                        candidate_count=len(candidates),
                        error=retrieval_error,
                    )
                )

    summary = summarize(rows)
    summary_path = report_dir / "rerank_eval_summary.json"
    details_path = report_dir / "rerank_eval_details.csv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_details_csv(details_path, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval with and without rerank.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--document-storage-dir", default=None)
    parser.add_argument("--chroma-persist-dir", default=None)
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional corpus file to index before evaluation.",
    )
    parser.add_argument(
        "--reset-storage",
        action="store_true",
        help="Reset the auto-created storage under report-dir before indexing corpus.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vector", "bm25", "hybrid"],
        choices=["vector", "bm25", "hybrid", "auto"],
    )
    parser.add_argument(
        "--rerank-modes",
        nargs="+",
        default=["none", "keyword"],
        choices=["none", "keyword", "semantic", "auto"],
    )
    parser.add_argument(
        "--warmup-reranker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm up semantic reranker before measured evaluation.",
    )
    return parser.parse_args()


def build_rerankers(rerank_modes: list[str]) -> dict[str, object]:
    rerankers: dict[str, object] = {}
    if "keyword" in rerank_modes:
        rerankers["keyword"] = KeywordOverlapReranker()
    if "semantic" in rerank_modes or "auto" in rerank_modes:
        rerankers["semantic"] = SemanticCrossEncoderReranker()
    return rerankers


def resolve_rerank_mode(
    rerank_mode: str,
    retrieval_mode: str,
    candidates: list,
    document_type,
) -> str:
    if rerank_mode != "auto":
        return rerank_mode
    return choose_auto_rerank_mode(
        retrieval_mode=retrieval_mode,
        results=candidates,
        document_type=document_type,
    )


def resolve_retrieval_mode(
    retrieval_mode: str,
    document_type,
) -> str:
    if retrieval_mode != "auto":
        return retrieval_mode
    return choose_auto_retrieval_mode(document_type=document_type)


def warmup_semantic_reranker(reranker: object) -> None:
    from app.retrieval.base import RetrievedChunk

    chunk = RetrievedChunk(
        id="warmup_chunk",
        document_id="warmup_document",
        filename="warmup.txt",
        chunk_index=0,
        text="This short passage is used to warm up semantic reranker inference.",
        score=1.0,
        retrieval_method="warmup",
    )
    reranker.score("warm up semantic reranker", [chunk])


def configure_corpus_storage(args: argparse.Namespace, report_dir: Path) -> None:
    storage_root = report_dir / "_storage"
    if args.reset_storage and storage_root.exists():
        ensure_child_path(storage_root, report_dir)
        shutil.rmtree(storage_root)

    if not args.document_storage_dir:
        args.document_storage_dir = str(storage_root / "documents")
    if not args.chroma_persist_dir:
        args.chroma_persist_dir = str(storage_root / "chroma")


async def ensure_corpus_indexed(
    document_service: DocumentService,
    corpus_path: Path,
) -> None:
    documents = await document_service.list_documents()
    if documents:
        return

    upload = UploadFile(
        file=BytesIO(corpus_path.read_bytes()),
        filename=corpus_path.name,
    )
    document = await document_service.upload(upload)
    if document.index_error:
        raise RuntimeError(f"Failed to index corpus {corpus_path}: {document.index_error}")
    if document.chunk_count <= 0:
        raise RuntimeError(f"Corpus {corpus_path} did not produce indexed chunks.")


def ensure_child_path(child: Path, parent: Path) -> None:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    if parent_resolved not in child_resolved.parents:
        raise ValueError(f"Refusing to delete path outside report dir: {child_resolved}")


def load_cases(path: Path) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                cases.append(
                    RetrievalEvalCase(
                        id=str(payload["id"]),
                        query=str(payload["query"]),
                        answers=[str(answer) for answer in payload["answers"]],
                        expected_document=payload.get("expected_document"),
                        tags=payload.get("tags"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid case at {path}:{line_number}: {exc}") from exc

    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def build_row(
    *,
    case: RetrievalEvalCase,
    mode: str,
    rerank_mode: str,
    results: list,
    retrieval_ms: float,
    rerank_ms: float,
    candidate_count: int,
    error: str | None,
) -> RerankEvalRow:
    metrics = evaluate_retrieved_chunks(results, case.answers)
    return RerankEvalRow(
        case_id=case.id,
        query=case.query,
        mode=mode,
        rerank_mode=rerank_mode,
        hit_at_1=metrics.hit_at_1,
        hit_at_3=metrics.hit_at_3,
        hit_at_5=metrics.hit_at_5,
        first_hit_rank=metrics.first_hit_rank,
        mrr=metrics.mrr,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        total_ms=retrieval_ms + rerank_ms,
        candidate_count=candidate_count,
        top_sources=format_top_sources(results),
        error=error,
    )


def error_row(case: RetrievalEvalCase, mode: str, error: str) -> RerankEvalRow:
    return RerankEvalRow(
        case_id=case.id,
        query=case.query,
        mode=mode,
        rerank_mode="none",
        hit_at_1=False,
        hit_at_3=False,
        hit_at_5=False,
        first_hit_rank=None,
        mrr=0.0,
        retrieval_ms=0.0,
        rerank_ms=0.0,
        total_ms=0.0,
        candidate_count=0,
        top_sources="",
        error=error,
    )


def summarize(rows: list[RerankEvalRow]) -> dict[str, dict[str, float]]:
    groups = sorted({(row.mode, row.rerank_mode) for row in rows})
    summary: dict[str, dict[str, float]] = {}
    for mode, rerank_mode in groups:
        group_rows = [
            row for row in rows if row.mode == mode and row.rerank_mode == rerank_mode
        ]
        if not group_rows:
            continue
        label = f"{mode}_{rerank_mode}"
        summary[label] = {
            "case_count": float(len(group_rows)),
            "hit_at_1": average(row.hit_at_1 for row in group_rows),
            "hit_at_3": average(row.hit_at_3 for row in group_rows),
            "hit_at_5": average(row.hit_at_5 for row in group_rows),
            "mrr": average(row.mrr for row in group_rows),
            "avg_retrieval_ms": average(row.retrieval_ms for row in group_rows),
            "avg_rerank_ms": average(row.rerank_ms for row in group_rows),
            "avg_total_ms": average(row.total_ms for row in group_rows),
            "avg_candidate_count": average(row.candidate_count for row in group_rows),
            "error_rate": average(row.error is not None for row in group_rows),
        }
    return summary


def write_details_csv(path: Path, rows: list[RerankEvalRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


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


if __name__ == "__main__":
    asyncio.run(main())
