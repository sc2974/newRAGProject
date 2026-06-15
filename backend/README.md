# Backend

FastAPI service skeleton for the RAG demo.

## Endpoints

- `GET /api/health`
- `POST /api/documents/upload`
- `GET /api/documents`
- `GET /api/documents/{document_id}`
- `GET /api/documents/{document_id}/content`
- `GET /api/documents/{document_id}/chunks`
- `POST /api/documents/search`
- `POST /api/documents/reindex`
- `DELETE /api/documents/{document_id}`
- `POST /api/llm/ask`
- `POST /api/rag/ask`
- `POST /api/chat/sessions`
- `GET /api/chat/sessions`
- `GET /api/chat/sessions/{session_id}`
- `POST /api/chat/sessions/{session_id}/messages`
- `DELETE /api/chat/sessions/{session_id}`
- `POST /api/notes`
- `GET /api/notes`
- `GET /api/notes/{note_id}`
- `PUT /api/notes/{note_id}`
- `DELETE /api/notes/{note_id}`

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Ollama

The current demo uses Ollama for both generation and embeddings:

- Generation model: `qwen2.5:0.5b`
- Embedding model: `qwen3-embedding:0.6b`

After changing the embedding model or chunking strategy, call:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/documents/reindex -Method Post
```
