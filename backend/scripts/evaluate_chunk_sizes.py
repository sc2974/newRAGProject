import argparse
import asyncio
import csv
import json
import shutil
from io import BytesIO
from pathlib import Path
import sys

from starlette.datastructures import UploadFile


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.evaluation.retrieval_evaluator import (
    RetrievalEvalCase,
    evaluate_retrievers,
    row_to_dict,
    summarize_rows,
)
from app.evaluation.retrieval_metrics import first_matching_rank
from app.retrieval.factory import build_retrievers, load_active_chunks
from app.services.document_service import DocumentService
from app.services.text_splitter import TextSplitter


DEFAULT_DATASETS = [
    {
        "name": "faq_rewrite",
        "corpus": BACKEND_DIR / "eval_data" / "corpora" / "customer_support_faq_corpus.md",
        "eval": BACKEND_DIR / "eval_data" / "faq" / "customer_support_faq_rewrite_eval.jsonl",
    },
    {
        "name": "legal_clauses",
        "corpus": BACKEND_DIR / "eval_data" / "corpora" / "legal_clauses_corpus.md",
        "eval": BACKEND_DIR / "eval_data" / "legal" / "legal_clauses_eval.jsonl",
    },
    {
        "name": "domain_scifact",
        "corpus": BACKEND_DIR / "eval_data" / "domain_scifact" / "corpora" / "beir_scifact_corpus.md",
        "eval": BACKEND_DIR / "eval_data" / "domain_scifact" / "beir_scifact_eval_80.jsonl",
    },
]


async def main() -> None:
    args = parse_args()
    report_root = Path(args.report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    all_summary_rows = []

    datasets = DEFAULT_DATASETS if not args.dataset else [parse_dataset_arg(args.dataset)]
    for dataset in datasets:
        dataset_rows = await run_dataset_experiment(dataset, args, report_root)
        all_summary_rows.extend(dataset_rows)

    write_summary_csv(report_root / "chunk_size_experiment_summary.csv", all_summary_rows)
    (report_root / "chunk_size_experiment_summary.json").write_text(
        json.dumps(all_summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(all_summary_rows, ensure_ascii=False, indent=2))


async def run_dataset_experiment(dataset: dict, args, report_root: Path) -> list[dict]:
    cases = load_cases(Path(dataset["eval"]))
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    dataset_dir = report_root / dataset["name"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    original_document_storage_dir = settings.document_storage_dir
    original_chroma_persist_dir = settings.chroma_persist_dir

    for chunk_size in args.chunk_sizes:
        chunk_overlap = int(chunk_size * args.overlap_ratio)
        run_name = f"chunk_{chunk_size}_overlap_{chunk_overlap}"
        storage_dir = report_root / "_storage" / dataset["name"] / run_name
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

        settings.document_storage_dir = str(storage_dir / "documents")
        settings.chroma_persist_dir = str(storage_dir / "chroma")

        document_service = DocumentService()
        document_service._text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        corpus_path = Path(dataset["corpus"])
        document = await document_service.upload(
            UploadFile(file=BytesIO(corpus_path.read_bytes()), filename=corpus_path.name)
        )
        chunks = await load_active_chunks(document_service)
        retrievers = await build_retrievers(document_service)
        eval_rows = evaluate_retrievers(
            cases,
            retrievers,
            top_k=args.top_k,
            output_modes=args.modes,
        )

        detail_rows = add_chunk_metrics(eval_rows, cases, retrievers, args.top_k)
        summary = summarize_rows(eval_rows)
        enriched_summary = enrich_summary(
            summary,
            detail_rows,
            dataset_name=dataset["name"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_count=document.chunk_count,
            chunks=chunks,
        )
        rows.extend(enriched_summary)

        run_dir = dataset_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "retrieval_eval_summary.json").write_text(
            json.dumps(enriched_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_details_csv(run_dir / "retrieval_eval_details.csv", detail_rows)

    settings.document_storage_dir = original_document_storage_dir
    settings.chroma_persist_dir = original_chroma_persist_dir
    return rows


def add_chunk_metrics(eval_rows, cases, retrievers, top_k: int) -> list[dict]:
    case_by_id = {case.id: case for case in cases}
    detail_rows = []
    for row in eval_rows:
        case = case_by_id[row.case_id]
        retriever = retrievers[row.mode]
        results = retriever.search(case.query, top_k)
        first_rank = first_matching_rank(results, case.answers)
        first_hit_chunk_chars = None
        answer_density = None
        if first_rank is not None:
            hit_chunk = results[first_rank - 1]
            first_hit_chunk_chars = len(hit_chunk.text)
            answer_density = max(len(answer) for answer in case.answers) / max(
                first_hit_chunk_chars,
                1,
            )

        detail = row_to_dict(row)
        detail["retrieved_context_chars"] = sum(len(result.text) for result in results)
        detail["first_hit_chunk_chars"] = first_hit_chunk_chars
        detail["answer_density"] = answer_density
        detail_rows.append(detail)
    return detail_rows


def enrich_summary(
    summary: dict,
    detail_rows: list[dict],
    *,
    dataset_name: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_count: int,
    chunks,
) -> list[dict]:
    enriched = []
    avg_chunk_chars = average(len(chunk.text) for chunk in chunks)
    for mode, metrics in summary.items():
        mode_rows = [row for row in detail_rows if row["mode"] == mode]
        enriched.append(
            {
                "dataset": dataset_name,
                "mode": mode,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_count": chunk_count,
                "avg_chunk_chars": avg_chunk_chars,
                **metrics,
                "avg_retrieved_context_chars": average(
                    row["retrieved_context_chars"] for row in mode_rows
                ),
                "avg_first_hit_chunk_chars": average(
                    row["first_hit_chunk_chars"]
                    for row in mode_rows
                    if row["first_hit_chunk_chars"] is not None
                ),
                "avg_answer_density": average(
                    row["answer_density"]
                    for row in mode_rows
                    if row["answer_density"] is not None
                ),
            }
        )
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chunk size impact.")
    parser.add_argument("--report-dir", default=str(BACKEND_DIR / "reports" / "chunk_size_experiments"))
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[300, 500, 800, 1200])
    parser.add_argument("--overlap-ratio", type=float, default=0.18)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=["vector", "bm25", "hybrid"])
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional custom dataset in name|corpus_path|eval_path format.",
    )
    return parser.parse_args()


def parse_dataset_arg(value: str) -> dict:
    name, corpus, eval_path = value.split("|", 2)
    return {"name": name, "corpus": Path(corpus), "eval": Path(eval_path)}


def load_cases(path: Path) -> list[RetrievalEvalCase]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
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
    return cases


def write_details_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average(values) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


if __name__ == "__main__":
    asyncio.run(main())
