import argparse
import json
from pathlib import Path
import sys
import time

import httpx


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.services.embedding_service import OllamaEmbeddingService


def main() -> None:
    args = parse_args()
    queries = load_queries(Path(args.dataset))
    embedding_service = OllamaEmbeddingService()

    pending = [
        query
        for query in queries
        if not embedding_service._cache_path(query).exists()
    ]
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "total_queries": len(queries),
                "cached_queries": len(queries) - len(pending),
                "pending_queries": len(pending),
                "batch_size": args.batch_size,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if not pending:
        return

    with httpx.Client(timeout=args.timeout) as client:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            for attempt in range(1, args.retries + 1):
                try:
                    response = client.post(
                        f"{settings.ollama_base_url.rstrip('/')}/api/embed",
                        json={
                            "model": settings.ollama_embedding_model,
                            "input": batch,
                        },
                    )
                    response.raise_for_status()
                    embeddings = response.json().get("embeddings") or []
                    if len(embeddings) != len(batch):
                        raise ValueError(
                            f"Expected {len(batch)} embeddings, got {len(embeddings)}"
                        )

                    for query, embedding in zip(batch, embeddings):
                        embedding_service._write_cache(
                            embedding_service._cache_path(query),
                            [float(value) for value in embedding],
                        )
                    print(
                        f"cached {min(start + len(batch), len(pending))}/{len(pending)}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    if attempt >= args.retries:
                        print(f"failed batch at {start}: {exc}", flush=True)
                        break
                    time.sleep(attempt * args.retry_sleep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute query embeddings.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2)
    return parser.parse_args()


def load_queries(path: Path) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            query = str(json.loads(line)["query"]).strip()
            if query and query not in seen:
                queries.append(query)
                seen.add(query)
    return queries


if __name__ == "__main__":
    main()
