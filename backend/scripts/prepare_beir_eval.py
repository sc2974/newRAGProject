import argparse
import csv
import json
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


DEFAULT_BEIR_DIR = PROJECT_DIR / "data" / "public" / "beir"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "eval_data" / "beir"
DEFAULT_CORPUS_DIR = BACKEND_DIR / "eval_data" / "corpora"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    corpus_dir = Path(args.corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for dataset in args.datasets:
        result = prepare_dataset(
            Path(args.beir_dir) / dataset,
            dataset=dataset,
            output_dir=output_dir,
            corpus_dir=corpus_dir,
            query_limit=args.query_limit,
            distractor_limit=args.distractor_limit,
        )
        manifest[dataset] = result

    manifest_path = output_dir / "beir_eval_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BEIR retrieval eval data.")
    parser.add_argument("--beir-dir", default=str(DEFAULT_BEIR_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fiqa", "quora"])
    parser.add_argument("--query-limit", type=int, default=500)
    parser.add_argument("--distractor-limit", type=int, default=1000)
    return parser.parse_args()


def prepare_dataset(
    dataset_dir: Path,
    *,
    dataset: str,
    output_dir: Path,
    corpus_dir: Path,
    query_limit: int,
    distractor_limit: int,
) -> dict:
    corpus = load_jsonl_by_id(dataset_dir / "corpus.jsonl")
    queries = load_jsonl_by_id(dataset_dir / "queries.jsonl")
    qrels_path = pick_qrels(dataset_dir)
    qrels = load_qrels(qrels_path)

    cases = []
    needed_doc_ids: set[str] = set()
    for query_id, doc_ids in qrels.items():
        if len(cases) >= query_limit:
            break
        query = queries.get(query_id)
        if not query:
            continue
        existing_doc_ids = [doc_id for doc_id in doc_ids if doc_id in corpus]
        if not existing_doc_ids:
            continue

        needed_doc_ids.update(existing_doc_ids)
        cases.append(
            {
                "id": f"{dataset}_{len(cases) + 1:04d}",
                "query": query.get("text", ""),
                "answers": [doc_answer_token(doc_id) for doc_id in existing_doc_ids],
                "expected_document": f"beir_{dataset}_corpus.md",
                "tags": ["en", "beir", dataset, dataset_type(dataset)],
                "group_id": f"{dataset}:{query_id}",
                "context_id": ",".join(existing_doc_ids[:5]),
                "source_query_id": query_id,
                "relevant_doc_ids": existing_doc_ids,
            }
        )

    distractor_doc_ids = [
        doc_id for doc_id in corpus.keys() if doc_id not in needed_doc_ids
    ][:distractor_limit]
    selected_doc_ids = list(needed_doc_ids) + distractor_doc_ids

    eval_path = output_dir / f"beir_{dataset}_eval_{len(cases)}.jsonl"
    corpus_path = corpus_dir / f"beir_{dataset}_corpus.md"
    write_jsonl(eval_path, cases)
    write_corpus(corpus_path, corpus, selected_doc_ids)

    return {
        "dataset": dataset,
        "type": dataset_type(dataset),
        "qrels": str(qrels_path),
        "cases": len(cases),
        "corpus_documents": len(selected_doc_ids),
        "eval_path": str(eval_path),
        "corpus_path": str(corpus_path),
    }


def pick_qrels(dataset_dir: Path) -> Path:
    for split in ("test", "dev", "train"):
        path = dataset_dir / "qrels" / f"{split}.tsv"
        if path.exists():
            return path
    raise FileNotFoundError(f"No qrels file found in {dataset_dir}")


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            payload = json.loads(line)
            records[str(payload["_id"])] = payload
    return records


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            score = int(row.get("score", row.get("relevance-score", "0")))
            if score <= 0:
                continue
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            qrels.setdefault(query_id, []).append(corpus_id)
    return qrels


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_corpus(path: Path, corpus: dict[str, dict], doc_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for doc_id in doc_ids:
            doc = corpus[doc_id]
            title = doc.get("title") or "untitled"
            text = doc.get("text") or ""
            file.write(f"# DOCID:{doc_id} | {title}\n\n")
            file.write(f"{doc_answer_token(doc_id)}\n\n")
            file.write(text.strip())
            file.write("\n\n")


def doc_answer_token(doc_id: str) -> str:
    return f"DOCID:{doc_id}"


def dataset_type(dataset: str) -> str:
    return {
        "scifact": "scientific_claim",
        "fiqa": "financial_qa",
        "quora": "semantic_question_match",
    }.get(dataset, "retrieval")


if __name__ == "__main__":
    main()
