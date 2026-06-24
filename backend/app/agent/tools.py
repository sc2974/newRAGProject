import base64
import json
import re
from collections import Counter
from datetime import datetime, timezone

from langchain_core.tools import tool

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.rag.service import RagService
from app.schemas.notes import NoteCreate
from app.schemas.rag import AskRequest
from app.services.document_service import DocumentService
from app.services.note_service import note_service
from app.services.vector_store import VectorStore


MAX_PREVIEW_CHARS = 420
MAX_NOTE_PREVIEW_CHARS = 360


def _preview(text: str, limit: int = MAX_PREVIEW_CHARS) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _note_text(note) -> str:
    tags = " ".join(note.tags or [])
    return f"{note.title}\n{tags}\n{note.content}"


def _score_note(query: str, note) -> tuple[float, dict[str, int]]:
    query_tokens = _tokenize_text(query)
    note_text = _note_text(note).lower()
    title_text = note.title.lower()
    tag_text = " ".join(note.tags or []).lower()
    content_text = note.content.lower()

    detail = {"title": 0, "tags": 0, "content": 0}
    score = 0.0
    for token in query_tokens:
        if not token:
            continue
        if token in title_text:
            detail["title"] += 1
            score += 5.0
        if token in tag_text:
            detail["tags"] += 1
            score += 3.0
        content_hits = content_text.count(token)
        if content_hits:
            detail["content"] += content_hits
            score += min(content_hits, 5)
    return score, detail


def _decode_jwt_payload_without_verification(token: str) -> dict | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{payload}{padding}".encode("utf-8"))
        return json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None


async def _active_chunks():
    document_service = DocumentService()
    documents = await document_service.list_documents()
    chunks = []
    for document in documents:
        document_chunks = await document_service.list_chunks(document.id)
        if document_chunks:
            chunks.extend(document_chunks)
    return documents, chunks


@tool(description="Get the current local date and time.")
async def what_time_is_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(
    description=(
        "Parse a JWT token and return user id/name information. This mirrors the original "
        "project's get_user_info_tools. In this demo the token payload is decoded without "
        "signature verification, so use it only for displaying debug information."
    )
)
async def get_user_info_tools(token: str) -> str:
    payload = _decode_jwt_payload_without_verification(token)
    token_preview = f"{token[:10]}...{token[-6:]}" if len(token) > 20 else token
    if not payload:
        return (
            "User info tool report\n"
            "- Tool compatibility: original project used Django JWT decoding.\n"
            "- Demo behavior: token signature verification is not configured.\n"
            f"- Token preview: {token_preview}\n"
            f"- Token length: {len(token)}\n"
            "- Result: token payload could not be decoded as JWT."
        )

    user_id = (
        payload.get("user_id")
        or payload.get("id")
        or payload.get("sub")
        or payload.get("uid")
        or "unknown"
    )
    user_name = (
        payload.get("user_name")
        or payload.get("username")
        or payload.get("name")
        or "unknown"
    )
    safe_payload = {
        key: value
        for key, value in payload.items()
        if key.lower() not in {"password", "secret", "token", "access_token"}
    }
    return (
        "User info tool report\n"
        "- Tool compatibility: mirrors original get_user_info_tools.\n"
        "- Demo behavior: decoded JWT payload without verifying signature.\n"
        f"- Token preview: {token_preview}\n"
        f"- User ID: {user_id}\n"
        f"- User name: {user_name}\n"
        f"- Payload keys: {', '.join(safe_payload.keys()) or 'none'}\n"
        f"- Safe payload: {json.dumps(safe_payload, ensure_ascii=False)}"
    )


@tool(
    description=(
        "Search user notes by title, tags, and content. This mirrors the original "
        "project's search_notes_tool. The current demo uses an in-memory note store and "
        "keyword scoring instead of MySQL plus vector note search."
    )
)
async def search_notes_tool(query: str, top_k: int = 5) -> str:
    notes = await note_service.list_notes()
    safe_top_k = max(1, min(top_k, 20))
    if not notes:
        return (
            "Note search report\n"
            f"- Query: {query}\n"
            "- Note store: in-memory demo store\n"
            "- Total notes: 0\n"
            "- Result: no notes exist yet. Use create_note_tool to create one first."
        )

    scored = []
    for note in notes:
        score, detail = _score_note(query, note)
        if score > 0:
            scored.append((score, detail, note))

    scored.sort(key=lambda item: (item[0], item[2].updated_at), reverse=True)
    if not scored:
        return (
            "Note search report\n"
            f"- Query: {query}\n"
            f"- Total notes scanned: {len(notes)}\n"
            "- Ranking method: keyword hits in title/tags/content\n"
            "- Result: no keyword-matched notes found."
        )

    lines = [
        "Note search report",
        f"- Query: {query}",
        f"- Total notes scanned: {len(notes)}",
        f"- Matched notes: {len(scored)}",
        f"- Showing: {min(safe_top_k, len(scored))}",
        "- Ranking method: title hits weight 5, tag hits weight 3, content hits weight 1",
        "- Demo note store: in-memory; restart clears notes until persistence is added.",
        "",
    ]
    for rank, (score, detail, note) in enumerate(scored[:safe_top_k], start=1):
        lines.extend(
            [
                f"Rank {rank}",
                f"  id: {note.id}",
                f"  title: {note.title}",
                f"  score: {score:.2f}",
                f"  hit_detail: title={detail['title']}, tags={detail['tags']}, content={detail['content']}",
                f"  tags: {', '.join(note.tags) if note.tags else '(none)'}",
                f"  created_at: {note.created_at.isoformat()}",
                f"  updated_at: {note.updated_at.isoformat()}",
                f"  preview: {_preview(note.content, MAX_NOTE_PREVIEW_CHARS)}",
                "",
            ]
        )
    return "\n".join(lines)


@tool(
    description=(
        "Return note statistics. This mirrors the original project's get_note_stats_tool. "
        "The original counted categories from MySQL; this demo counts in-memory notes "
        "and tag distribution."
    )
)
async def get_note_stats_tool() -> str:
    notes = await note_service.list_notes()
    tag_counter = Counter(tag for note in notes for tag in note.tags)
    total_chars = sum(len(note.content) for note in notes)
    if notes:
        newest = max(note.created_at for note in notes)
        latest_update = max(note.updated_at for note in notes)
    else:
        newest = None
        latest_update = None

    lines = [
        "Note statistics report",
        "- Tool compatibility: mirrors original get_note_stats_tool.",
        "- Demo difference: categories are not implemented yet; tags are counted instead.",
        f"- Total notes: {len(notes)}",
        f"- Total content characters: {total_chars}",
        f"- Average content length: {(total_chars / len(notes)):.1f}" if notes else "- Average content length: 0",
        f"- Newest note created_at: {newest.isoformat() if newest else 'none'}",
        f"- Latest note updated_at: {latest_update.isoformat() if latest_update else 'none'}",
        "",
        "Tag distribution:",
    ]
    if tag_counter:
        for tag, count in tag_counter.most_common(20):
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- No tags available.")
    return "\n".join(lines)


@tool(
    description=(
        "Return notes that should be reviewed today. This mirrors the original project's "
        "get_today_reviews_tool. The demo has no spaced-repetition scheduler yet, so it "
        "returns recent notes as review candidates with clear explanation."
    )
)
async def get_today_reviews_tool() -> str:
    notes = await note_service.list_notes()
    if not notes:
        return (
            "Today review report\n"
            "- Tool compatibility: mirrors original get_today_reviews_tool.\n"
            "- Demo difference: spaced-repetition scheduling is not implemented yet.\n"
            "- Result: no notes exist yet."
        )

    candidates = sorted(notes, key=lambda note: note.updated_at, reverse=True)[:8]
    lines = [
        "Today review report",
        "- Tool compatibility: mirrors original get_today_reviews_tool.",
        "- Demo difference: no Ebbinghaus/spaced-repetition table yet.",
        "- Selection rule: newest updated notes are listed as temporary review candidates.",
        f"- Candidate count: {len(candidates)}",
        "",
    ]
    for index, note in enumerate(candidates, start=1):
        lines.extend(
            [
                f"{index}. {note.title}",
                f"   id: {note.id}",
                f"   tags: {', '.join(note.tags) if note.tags else '(none)'}",
                f"   updated_at: {note.updated_at.isoformat()}",
                f"   preview: {_preview(note.content, 220)}",
                "",
            ]
        )
    return "\n".join(lines)


@tool(
    description=(
        "Mark a note as reviewed. This mirrors the original project's mark_reviewed_tool. "
        "The demo validates the note id and returns a simulated review result; persistent "
        "review scheduling will be added later with a database."
    )
)
async def mark_reviewed_tool(note_id: str) -> str:
    note = await note_service.get_note(note_id)
    if note is None:
        return (
            "Mark reviewed report\n"
            f"- note_id: {note_id}\n"
            "- Result: note not found in the in-memory demo store."
        )

    now = datetime.now(timezone.utc)
    return (
        "Mark reviewed report\n"
        "- Tool compatibility: mirrors original mark_reviewed_tool.\n"
        "- Demo difference: review state is not persisted yet.\n"
        f"- note_id: {note.id}\n"
        f"- title: {note.title}\n"
        f"- reviewed_at: {now.isoformat()}\n"
        "- simulated_next_interval_days: 1\n"
        "- Result: note exists and has been treated as reviewed for this Agent response."
    )


@tool(
    description=(
        "Create a user note with title and optional content. This mirrors the original "
        "project's create_note_tool. The demo stores notes in memory and supports tags "
        "only through the note API, so this tool creates title/content without auto tags."
    )
)
async def create_note_tool(title: str, content: str = "") -> str:
    note = await note_service.create_note(NoteCreate(title=title, content=content))
    return (
        "Create note report\n"
        "- Tool compatibility: mirrors original create_note_tool.\n"
        "- Demo difference: note is stored in memory; auto vector indexing/tagging is not implemented yet.\n"
        f"- id: {note.id}\n"
        f"- title: {note.title}\n"
        f"- content_length: {len(note.content)}\n"
        f"- tags: {', '.join(note.tags) if note.tags else '(none)'}\n"
        f"- created_at: {note.created_at.isoformat()}\n"
        f"- preview: {_preview(note.content, MAX_NOTE_PREVIEW_CHARS)}"
    )


@tool(
    description=(
        "Find notes related to a note id, and also include related knowledge-base chunks. "
        "This mirrors the original project's get_related_notes_tool. The demo ranks notes "
        "by token overlap and knowledge-base content by BM25."
    )
)
async def get_related_notes_tool(note_id: str, top_k: int = 3) -> str:
    target = await note_service.get_note(note_id)
    safe_top_k = max(1, min(top_k, 10))
    if target is None:
        return (
            "Related notes report\n"
            f"- note_id: {note_id}\n"
            "- Result: note not found in the in-memory demo store."
        )

    all_notes = await note_service.list_notes()
    target_tokens = set(_tokenize_text(_note_text(target)))
    related_notes = []
    for note in all_notes:
        if note.id == target.id:
            continue
        note_tokens = set(_tokenize_text(_note_text(note)))
        overlap = target_tokens & note_tokens
        if overlap:
            denominator = max(1, len(target_tokens | note_tokens))
            related_notes.append((len(overlap) / denominator, overlap, note))
    related_notes.sort(key=lambda item: item[0], reverse=True)

    documents, chunks = await _active_chunks()
    bm25_results = BM25Retriever(chunks).search(query=_note_text(target), limit=safe_top_k) if chunks else []

    lines = [
        "Related notes and knowledge-base report",
        "- Tool compatibility: mirrors original get_related_notes_tool.",
        "- Demo note ranking: token-overlap Jaccard score.",
        "- Demo knowledge-base ranking: BM25 over active chunks.",
        f"- Target note id: {target.id}",
        f"- Target title: {target.title}",
        f"- Target token count: {len(target_tokens)}",
        f"- Active documents scanned: {len(documents)}",
        f"- Active chunks scanned: {len(chunks)}",
        "",
        "Related notes:",
    ]
    if related_notes:
        for rank, (score, overlap, note) in enumerate(related_notes[:safe_top_k], start=1):
            sample_terms = ", ".join(sorted(overlap)[:12])
            lines.extend(
                [
                    f"{rank}. {note.title}",
                    f"   id: {note.id}",
                    f"   similarity: {score:.4f}",
                    f"   shared_terms: {sample_terms}",
                    f"   preview: {_preview(note.content, 220)}",
                    "",
                ]
            )
    else:
        lines.append("- No related notes found.")

    lines.extend(["", "Related knowledge-base chunks:"])
    if bm25_results:
        for rank, result in enumerate(bm25_results, start=1):
            score = "None" if result.score is None else f"{result.score:.4f}"
            lines.extend(
                [
                    f"{rank}. {result.filename}",
                    f"   document_id: {result.document_id}",
                    f"   chunk_index: {result.chunk_index}",
                    f"   score: {score}",
                    f"   preview: {_preview(result.text, 260)}",
                    "",
                ]
            )
    else:
        lines.append("- No knowledge-base chunks found.")
    return "\n".join(lines)


@tool(
    description=(
        "List uploaded active documents with metadata, document type, chunk count, "
        "index status, md5, and text preview. Use this before answering questions about "
        "what data is currently in the knowledge base."
    )
)
async def list_documents_tool() -> str:
    document_service = DocumentService()
    documents = await document_service.list_documents()
    if not documents:
        return "No active documents are currently available in the knowledge base."

    total_chunks = sum(document.chunk_count for document in documents)
    total_text_length = sum(document.text_length for document in documents)
    lines = [
        "Knowledge base document inventory:",
        f"- Active documents: {len(documents)}",
        f"- Total indexed chunks reported by metadata: {total_chunks}",
        f"- Total text length: {total_text_length} characters",
        "",
    ]
    for index, document in enumerate(documents, start=1):
        lines.extend(
            [
                f"{index}. {document.filename}",
                f"   id: {document.id}",
                f"   status: {document.status}",
                f"   type: {document.document_type} "
                f"(confidence {document.document_type_confidence:.3f})",
                f"   version: {document.version}",
                f"   chunks: {document.chunk_count}",
                f"   size_bytes: {document.size_bytes}",
                f"   text_length: {document.text_length}",
                f"   md5: {document.content_md5}",
                f"   preview: {_preview(document.text_preview, 260)}",
                "",
            ]
        )
    return "\n".join(lines)


@tool(
    description=(
        "Show chunks for one uploaded document. The document_id should come from "
        "list_documents_tool. Use this when the user wants to inspect how a document was "
        "split or wants to see chunk text."
    )
)
async def inspect_document_chunks_tool(document_id: str, limit: int = 5) -> str:
    document_service = DocumentService()
    document = await document_service.get_document(document_id)
    if document is None:
        return f"Document not found or inactive: {document_id}"

    chunks = await document_service.list_chunks(document_id)
    if not chunks:
        return f"Document {document.filename} has no chunks available."

    safe_limit = max(1, min(limit, 20))
    lines = [
        f"Chunk inspection for {document.filename}",
        f"- document_id: {document.id}",
        f"- reported chunk_count: {document.chunk_count}",
        f"- actual chunks returned: {len(chunks)}",
        f"- showing first {min(safe_limit, len(chunks))} chunks",
        "",
    ]
    for chunk in chunks[:safe_limit]:
        lines.extend(
            [
                f"chunk #{chunk.chunk_index + 1}",
                f"id: {chunk.id}",
                f"chars: {len(chunk.text)}",
                f"text: {_preview(chunk.text)}",
                "",
            ]
        )
    return "\n".join(lines)


@tool(
    description=(
        "Run vector similarity search only and return detailed ranked chunks. Use this "
        "to diagnose semantic retrieval behavior."
    )
)
async def vector_search_tool(query: str, limit: int = 5) -> str:
    safe_limit = max(1, min(limit, 10))
    results = VectorRetriever(VectorStore()).search(query=query, limit=safe_limit)
    return _format_retrieval_results("Vector search", query, results)


@tool(
    description=(
        "Run BM25 keyword search only and return detailed ranked chunks. Use this to "
        "diagnose exact keyword matching behavior."
    )
)
async def bm25_search_tool(query: str, limit: int = 5) -> str:
    safe_limit = max(1, min(limit, 10))
    documents, chunks = await _active_chunks()
    retriever = BM25Retriever(chunks)
    results = retriever.search(query=query, limit=safe_limit)
    header = [
        f"BM25 corpus summary: {len(documents)} active documents, {len(chunks)} chunks",
        "",
    ]
    return "\n".join(header) + _format_retrieval_results("BM25 search", query, results)


@tool(
    description=(
        "Compare vector, BM25, and hybrid retrieval for the same query. Use this when "
        "the user wants to understand which retrieval method hits better or why a RAG "
        "answer used certain chunks."
    )
)
async def compare_retrieval_modes_tool(query: str, limit: int = 5) -> str:
    safe_limit = max(1, min(limit, 10))
    documents, chunks = await _active_chunks()
    vector_retriever = VectorRetriever(VectorStore())
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

    vector_results = vector_retriever.search(query=query, limit=safe_limit)
    bm25_results = bm25_retriever.search(query=query, limit=safe_limit)
    hybrid_results = hybrid_retriever.search(query=query, limit=safe_limit)

    sections = [
        "Retrieval comparison report",
        f"Query: {query}",
        f"Corpus: {len(documents)} active documents, {len(chunks)} chunks",
        "",
        _format_retrieval_results("Vector", query, vector_results),
        "",
        _format_retrieval_results("BM25", query, bm25_results),
        "",
        _format_retrieval_results("Hybrid RRF", query, hybrid_results),
    ]
    return "\n".join(sections)


@tool(
    description=(
        "Search the uploaded knowledge base with the existing RAG pipeline and return "
        "a grounded answer with source snippets. Use this when the user asks about "
        "uploaded documents, knowledge-base content, or facts that should be grounded "
        "in retrieved chunks."
    )
)
async def rag_search_tool(query: str) -> str:
    rag_service = RagService()
    response = await rag_service.ask(
        AskRequest(
            query=query,
            retrieval_mode="auto",
            rerank_mode="auto",
            limit=4,
            candidate_limit=20,
            hyde=True,
        )
    )
    source_lines = []
    for source in response.sources[:4]:
        score = "None" if source.score is None else f"{source.score:.4f}"
        rerank_score = (
            "None" if source.rerank_score is None else f"{source.rerank_score:.4f}"
        )
        source_lines.extend(
            [
                f"{source.citation_label} {source.title}",
                f"   document_id: {source.document_id}",
                f"   chunk_index: {source.chunk_index}",
                f"   retrieval_method: {source.retrieval_method}",
                f"   score: {score}",
                f"   rerank_score: {rerank_score}",
                f"   preview: {_preview(source.content_preview)}",
                "",
            ]
        )

    sources = "\n".join(source_lines) if source_lines else "No sources returned."
    hyde_query = response.hyde_query or "No HyDE hypothesis generated."
    hyde_error = response.hyde_error or "None"
    hyde_ms = 0.0 if response.hyde_ms is None else response.hyde_ms
    return (
        "Detailed RAG execution report\n"
        f"- Query: {response.query}\n"
        f"- HyDE enabled: {response.hyde}\n"
        f"- HyDE generation time: {hyde_ms:.1f} ms\n"
        f"- HyDE error: {hyde_error}\n"
        f"- HyDE hypothesis: {hyde_query}\n"
        f"- RAG status: {response.retrieval_status}\n"
        f"- Used retrieval: {response.used_retrieval}\n"
        f"- Retrieval mode: {response.retrieval_mode}\n"
        f"- Rerank enabled: {response.rerank}\n"
        f"- Rerank mode: {response.rerank_mode}\n"
        f"- Max score: {response.max_score}\n"
        f"- Score threshold: {response.score_threshold}\n"
        f"- Retrieval time: {response.retrieval_ms:.1f} ms\n"
        f"- Rerank time: {response.rerank_ms:.1f} ms\n"
        f"- Total retrieval time: {response.total_retrieval_ms:.1f} ms\n"
        f"- Generated by: {response.generated_by}\n"
        f"- Answer: {response.answer}\n\n"
        f"Sources:\n{sources}"
    )


@tool(
    description=(
        "Search the uploaded knowledge base with HyDE enabled. HyDE first asks the LLM "
        "to write a short hypothetical source passage for the user's question, then uses "
        "that passage plus the original query for retrieval. Use this when the user asks "
        "for HyDE, enhanced semantic retrieval, vague questions, or when ordinary RAG may "
        "miss semantically related chunks."
    )
)
async def hyde_rag_search_tool(query: str) -> str:
    rag_service = RagService()
    response = await rag_service.ask(
        AskRequest(
            query=query,
            retrieval_mode="auto",
            rerank_mode="auto",
            limit=4,
            candidate_limit=20,
            hyde=True,
        )
    )
    source_lines = []
    for source in response.sources[:4]:
        score = "None" if source.score is None else f"{source.score:.4f}"
        rerank_score = (
            "None" if source.rerank_score is None else f"{source.rerank_score:.4f}"
        )
        source_lines.extend(
            [
                f"{source.citation_label} {source.title}",
                f"   document_id: {source.document_id}",
                f"   chunk_index: {source.chunk_index}",
                f"   retrieval_method: {source.retrieval_method}",
                f"   score: {score}",
                f"   rerank_score: {rerank_score}",
                f"   preview: {_preview(source.content_preview)}",
                "",
            ]
        )

    sources = "\n".join(source_lines) if source_lines else "No sources returned."
    hyde_query = response.hyde_query or "No HyDE hypothesis generated."
    hyde_error = response.hyde_error or "None"
    hyde_ms = 0.0 if response.hyde_ms is None else response.hyde_ms
    return (
        "Detailed HyDE RAG execution report\n"
        f"- Query: {response.query}\n"
        f"- HyDE enabled: {response.hyde}\n"
        f"- HyDE generation time: {hyde_ms:.1f} ms\n"
        f"- HyDE error: {hyde_error}\n"
        f"- HyDE hypothesis: {hyde_query}\n"
        f"- RAG status: {response.retrieval_status}\n"
        f"- Used retrieval: {response.used_retrieval}\n"
        f"- Retrieval mode: {response.retrieval_mode}\n"
        f"- Rerank enabled: {response.rerank}\n"
        f"- Rerank mode: {response.rerank_mode}\n"
        f"- Max score: {response.max_score}\n"
        f"- Score threshold: {response.score_threshold}\n"
        f"- Retrieval time: {response.retrieval_ms:.1f} ms\n"
        f"- Rerank time: {response.rerank_ms:.1f} ms\n"
        f"- Total retrieval time: {response.total_retrieval_ms:.1f} ms\n"
        f"- Generated by: {response.generated_by}\n"
        f"- Answer: {response.answer}\n\n"
        f"Sources:\n{sources}"
    )


def _format_retrieval_results(title: str, query: str, results) -> str:
    lines = [
        f"{title} results",
        f"Query: {query}",
        f"Returned chunks: {len(results)}",
    ]
    if not results:
        lines.append("No chunks matched.")
        return "\n".join(lines)

    for rank, result in enumerate(results, start=1):
        score = "None" if result.score is None else f"{result.score:.4f}"
        lines.extend(
            [
                "",
                f"Rank {rank}",
                f"  file: {result.filename}",
                f"  document_id: {result.document_id}",
                f"  chunk_index: {result.chunk_index}",
                f"  method: {result.retrieval_method}",
                f"  score: {score}",
                f"  metadata: {result.metadata}",
                f"  text: {_preview(result.text)}",
            ]
        )
    return "\n".join(lines)


DEFAULT_AGENT_TOOLS = [
    what_time_is_now,
    get_user_info_tools,
    list_documents_tool,
    inspect_document_chunks_tool,
    vector_search_tool,
    bm25_search_tool,
    compare_retrieval_modes_tool,
    rag_search_tool,
    hyde_rag_search_tool,
]
