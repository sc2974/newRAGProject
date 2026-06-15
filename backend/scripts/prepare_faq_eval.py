import argparse
import json
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent


def main() -> None:
    args = parse_args()
    source_paths = [Path(path) for path in args.sources]
    rows = []
    for path in source_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("rows", []))

    pairs = []
    seen_questions: set[str] = set()
    for item in rows:
        row = item.get("row", {})
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer or question in seen_questions:
            continue
        seen_questions.add(question)
        pairs.append((question, answer))

    pairs = pairs[: args.limit]
    output_dir = Path(args.output_dir)
    corpus_dir = Path(args.corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpus_dir / args.corpus_name
    eval_path = output_dir / args.eval_name

    write_corpus(corpus_path, pairs)
    write_eval(eval_path, pairs)

    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "pairs": len(pairs),
                "corpus_path": str(corpus_path),
                "eval_path": str(eval_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FAQ eval data.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            str(PROJECT_DIR / "data" / "public" / "faq" / "customer_support_faq_rows_0_100.json"),
            str(PROJECT_DIR / "data" / "public" / "faq" / "customer_support_faq_rows_100_200.json"),
        ],
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output-dir", default=str(BACKEND_DIR / "eval_data" / "faq"))
    parser.add_argument("--corpus-dir", default=str(BACKEND_DIR / "eval_data" / "corpora"))
    parser.add_argument("--corpus-name", default="customer_support_faq_corpus.md")
    parser.add_argument("--eval-name", default="customer_support_faq_eval.jsonl")
    return parser.parse_args()


def write_corpus(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for index, (question, answer) in enumerate(pairs, start=1):
            doc_id = f"faq_{index:04d}"
            file.write(f"# DOCID:{doc_id}\n\n")
            file.write(f"Q: {question}\n")
            file.write(f"A: {answer}\n\n")


def write_eval(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for index, (question, answer) in enumerate(pairs, start=1):
            doc_id = f"faq_{index:04d}"
            row = {
                "id": f"faq_{index:04d}",
                "query": question,
                "answers": [answer],
                "expected_document": "customer_support_faq_corpus.md",
                "tags": ["en", "faq", "question_answer"],
                "group_id": doc_id,
                "context_id": doc_id,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
