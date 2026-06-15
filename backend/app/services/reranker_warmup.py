import logging
from threading import Thread

from app.core.config import settings
from app.retrieval.base import RetrievedChunk
from app.retrieval.reranker import SemanticCrossEncoderReranker


logger = logging.getLogger(__name__)
_semantic_reranker = SemanticCrossEncoderReranker()
_started = False


def get_semantic_reranker() -> SemanticCrossEncoderReranker:
    return _semantic_reranker


def start_semantic_reranker_warmup() -> None:
    global _started
    if _started or not settings.semantic_reranker_warmup:
        return

    _started = True
    thread = Thread(target=_warmup, name="semantic-reranker-warmup", daemon=True)
    thread.start()


def _warmup() -> None:
    try:
        chunk = RetrievedChunk(
            id="warmup_chunk",
            document_id="warmup_document",
            filename="warmup.txt",
            chunk_index=0,
            text="This short passage is used to warm up the semantic reranker.",
            score=1.0,
            retrieval_method="warmup",
        )
        _semantic_reranker.score("warm up semantic reranker", [chunk])
        logger.info("Semantic reranker warmed up on %s", _semantic_reranker.device)
    except Exception:
        logger.exception("Semantic reranker warmup failed")
