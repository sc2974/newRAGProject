import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


DEFAULT_CMRC_SOURCE = PROJECT_DIR / "data" / "public" / "cmrc2018" / "cmrc2018_dev.json"
DEFAULT_SQUAD_SOURCE = PROJECT_DIR / "data" / "public" / "squad-v1.1" / "dev-v1.1.json"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "eval_data"
DEFAULT_CORPUS_DIR = BACKEND_DIR / "eval_data" / "corpora"


@dataclass
class ParagraphRecord:
    dataset: str
    language: str
    title: str
    context: str
    paragraph_index: int
    qas: list[dict]

    @property
    def group_id(self) -> str:
        return f"{self.dataset}:p{self.paragraph_index}"

    @property
    def context_id(self) -> str:
        return self.group_id


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    corpus_dir = Path(args.corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    cmrc_records = load_squad_style(
        Path(args.cmrc_source),
        dataset="cmrc2018_dev",
        language="zh",
    )
    squad_records = load_squad_style(
        Path(args.squad_source),
        dataset="squad_v1_1_dev",
        language="en",
    )

    cmrc_cases, cmrc_contexts = build_cases(
        cmrc_records,
        limit=args.limit_per_language,
        expected_document="cmrc2018_dev_corpus.md",
    )
    squad_cases, squad_contexts = build_cases(
        squad_records,
        limit=args.limit_per_language,
        expected_document="squad_v1_1_dev_corpus.md",
    )

    mixed_cases = interleave_cases(cmrc_cases, squad_cases)
    similar_cases = build_similar_subset(
        cmrc_cases,
        squad_cases,
        limit=args.similar_limit,
    )

    write_jsonl(output_dir / f"cmrc2018_dev_eval_{len(cmrc_cases)}.jsonl", cmrc_cases)
    write_jsonl(output_dir / f"squad_v1_1_dev_eval_{len(squad_cases)}.jsonl", squad_cases)
    write_jsonl(output_dir / f"mixed_zh_en_eval_{len(mixed_cases)}.jsonl", mixed_cases)
    write_jsonl(
        output_dir / f"similar_questions_eval_{len(similar_cases)}.jsonl",
        similar_cases,
    )

    write_corpus(corpus_dir / "cmrc2018_dev_corpus.md", cmrc_contexts)
    write_corpus(corpus_dir / "squad_v1_1_dev_corpus.md", squad_contexts)

    manifest = {
        "cmrc_cases": len(cmrc_cases),
        "squad_cases": len(squad_cases),
        "mixed_cases": len(mixed_cases),
        "similar_cases": len(similar_cases),
        "cmrc_contexts": len(cmrc_contexts),
        "squad_contexts": len(squad_contexts),
        "outputs": {
            "cmrc": str(output_dir / f"cmrc2018_dev_eval_{len(cmrc_cases)}.jsonl"),
            "squad": str(output_dir / f"squad_v1_1_dev_eval_{len(squad_cases)}.jsonl"),
            "mixed": str(output_dir / f"mixed_zh_en_eval_{len(mixed_cases)}.jsonl"),
            "similar": str(output_dir / f"similar_questions_eval_{len(similar_cases)}.jsonl"),
            "cmrc_corpus": str(corpus_dir / "cmrc2018_dev_corpus.md"),
            "squad_corpus": str(corpus_dir / "squad_v1_1_dev_corpus.md"),
        },
    }
    (output_dir / "prepared_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare larger RAG evaluation datasets.")
    parser.add_argument("--cmrc-source", default=str(DEFAULT_CMRC_SOURCE))
    parser.add_argument("--squad-source", default=str(DEFAULT_SQUAD_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--limit-per-language", type=int, default=1000)
    parser.add_argument("--similar-limit", type=int, default=600)
    return parser.parse_args()


def load_squad_style(path: Path, *, dataset: str, language: str) -> list[ParagraphRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[ParagraphRecord] = []
    paragraph_index = 0
    for article in payload.get("data", []):
        title = str(article.get("title", "untitled"))
        for paragraph in article.get("paragraphs", []):
            qas = [
                qa
                for qa in paragraph.get("qas", [])
                if extract_answers(qa)
            ]
            if not qas:
                continue

            records.append(
                ParagraphRecord(
                    dataset=dataset,
                    language=language,
                    title=title,
                    context=str(paragraph.get("context", "")),
                    paragraph_index=paragraph_index,
                    qas=qas,
                )
            )
            paragraph_index += 1
    return records


def build_cases(
    records: list[ParagraphRecord],
    *,
    limit: int,
    expected_document: str,
) -> tuple[list[dict], list[ParagraphRecord]]:
    prioritized_records = sorted(records, key=lambda record: len(record.qas), reverse=True)
    cases: list[dict] = []
    used_contexts: dict[str, ParagraphRecord] = {}

    for record in prioritized_records:
        if len(cases) >= limit:
            break

        group_size = len(record.qas)
        for qa in record.qas:
            if len(cases) >= limit:
                break

            case_index = len(cases) + 1
            tags = [record.language, "qa"]
            if group_size > 1:
                tags.append("similar_group")

            cases.append(
                {
                    "id": f"{record.dataset}_{case_index:04d}",
                    "query": str(qa.get("question", "")).strip(),
                    "answers": extract_answers(qa),
                    "expected_document": expected_document,
                    "tags": tags,
                    "group_id": record.group_id,
                    "context_id": record.context_id,
                    "context_title": record.title,
                    "similar_group_size": group_size,
                }
            )
            used_contexts[record.context_id] = record

    return cases, list(used_contexts.values())


def build_similar_subset(
    cmrc_cases: list[dict],
    squad_cases: list[dict],
    *,
    limit: int,
) -> list[dict]:
    selected: list[dict] = []
    for cases in (cmrc_cases, squad_cases):
        groups: dict[str, list[dict]] = {}
        for case in cases:
            if "similar_group" not in case.get("tags", []):
                continue
            groups.setdefault(case["group_id"], []).append(case)

        for group_cases in groups.values():
            if len(group_cases) < 2:
                continue
            remaining = limit - len(selected)
            if remaining <= 0:
                break
            selected.extend(group_cases[:remaining])

    return selected[:limit]


def extract_answers(qa: dict) -> list[str]:
    answers = []
    for answer in qa.get("answers", []):
        text = str(answer.get("text", "")).strip()
        if text and text not in answers:
            answers.append(text)
    return answers


def interleave_cases(left: list[dict], right: list[dict]) -> list[dict]:
    mixed: list[dict] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            mixed.append(left[index])
        if index < len(right):
            mixed.append(right[index])
    return mixed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_corpus(path: Path, records: list[ParagraphRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(f"# {record.context_id} | {record.title}\n\n")
            file.write(record.context.strip())
            file.write("\n\n")


if __name__ == "__main__":
    main()
