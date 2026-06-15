import re
from dataclasses import dataclass
from pathlib import Path

from app.schemas.documents import DocumentType


QA_HEADER_PATTERN = re.compile(r"^#\s+[^|]+\|", re.MULTILINE)
FAQ_PATTERN = re.compile(r"\b(faq|question|answer|support|customer)\b", re.IGNORECASE)
LEGAL_PATTERN = re.compile(
    r"\b(section|article|clause|constitution|agreement|contract|party|liability|"
    r"confidentiality|termination|law|regulation)\b",
    re.IGNORECASE,
)
SCIENTIFIC_PATTERN = re.compile(
    r"\b(abstract|background|methods|results|conclusion|findings|funding|"
    r"doi|trial|study|patients|evidence)\b",
    re.IGNORECASE,
)
CODE_PATTERN = re.compile(
    r"^\s*(def|class|import|from|function|const|let|var)\s+[\w_]+",
    re.MULTILINE,
)
TABULAR_PATTERN = re.compile(r"[,|\t].+[,|\t]")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class DocumentTypePrediction:
    document_type: DocumentType
    confidence: float
    scores: dict[str, float]


def classify_document(filename: str, text: str) -> DocumentTypePrediction:
    sample = text[:20000]
    name = Path(filename).name.lower()
    scores = {
        "faq": 0.0,
        "legal": 0.0,
        "qa": 0.0,
        "scientific": 0.0,
        "code": 0.0,
        "tabular": 0.0,
        "general": 0.1,
    }

    scores["faq"] += 2.0 if FAQ_PATTERN.search(name) else 0.0
    scores["legal"] += 2.0 if LEGAL_PATTERN.search(name) else 0.0
    scores["scientific"] += 2.0 if SCIENTIFIC_PATTERN.search(name) else 0.0
    scores["qa"] += 1.5 if "cmrc" in name or "squad" in name else 0.0
    scores["code"] += 2.0 if Path(filename).suffix.lower() in {".py", ".js", ".ts"} else 0.0
    scores["tabular"] += 2.0 if Path(filename).suffix.lower() in {".csv", ".tsv"} else 0.0

    scores["qa"] += min(len(QA_HEADER_PATTERN.findall(sample)) / 8, 3.0)
    scores["faq"] += count_markers(sample, ["faq", "question", "answer", "refund", "password"])
    scores["legal"] += count_markers(
        sample,
        ["section", "article", "clause", "constitution", "agreement", "party"],
    )
    scores["scientific"] += count_markers(
        sample,
        ["background", "methods", "results", "conclusion", "findings", "funding"],
    )
    scores["code"] += min(len(CODE_PATTERN.findall(sample)) / 4, 2.0)
    scores["tabular"] += min(len(TABULAR_PATTERN.findall(sample)) / 20, 2.0)

    if CJK_PATTERN.search(sample) and QA_HEADER_PATTERN.search(sample):
        scores["qa"] += 1.0

    document_type = max(scores.items(), key=lambda item: item[1])[0]
    confidence = scores[document_type] / max(sum(scores.values()), 1.0)
    if scores[document_type] < 1.0:
        document_type = "general"
        confidence = 0.1

    return DocumentTypePrediction(
        document_type=document_type,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        scores={key: round(value, 3) for key, value in scores.items()},
    )


def count_markers(text: str, markers: list[str]) -> float:
    normalized = text.lower()
    return min(sum(normalized.count(marker) for marker in markers) / 10, 3.0)
