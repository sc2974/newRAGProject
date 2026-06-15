import argparse
import asyncio
import csv
import json
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.evaluation.retrieval_evaluator import (
    RetrievalEvalCase,
    evaluate_retrievers,
    row_to_dict,
    summarize_rows,
)
from app.core.config import settings
from app.retrieval.factory import build_retrievers
from app.services.document_service import DocumentService


DEFAULT_DATASET = BACKEND_DIR / "eval_data" / "current_rag_eval.jsonl"
DEFAULT_REPORT_DIR = BACKEND_DIR / "reports"


async def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(dataset_path)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    if args.embedding_timeout is not None:
        settings.ollama_timeout_seconds = args.embedding_timeout

    document_service = DocumentService()
    retrievers = await build_retrievers(document_service)

    rows = evaluate_retrievers(
        cases,
        retrievers,
        top_k=args.top_k,
        output_modes=args.modes,
    )
    summary = summarize_rows(rows)

    summary_path = report_dir / "retrieval_eval_summary.json"
    details_path = report_dir / "retrieval_eval_details.csv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_details_csv(details_path, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--embedding-timeout", type=float, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vector", "bm25", "hybrid"],
        choices=["vector", "bm25", "hybrid"],
    )
    return parser.parse_args()


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


def write_details_csv(path: Path, rows) -> None:
    fieldnames = [
        "case_id",
        "query",
        "mode",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "first_hit_rank",
        "mrr",
        "latency_ms",
        "top_sources",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_dict(row))


if __name__ == "__main__":
    asyncio.run(main())
