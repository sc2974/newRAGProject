# LangChain RAG FastAPI Demo

This workspace is a staged demo inspired by `RMA-MUN/LangChain-RAG-FastAPI-Service`.
The first milestone is a runnable skeleton. RAG, notebook, auth, and vector search
features will be implemented step by step later.

## Structure

```text
backend/
  app/
    api/          FastAPI routers
    core/         Settings and shared configuration
    rag/          RAG service placeholders
    schemas/      Pydantic request/response models
    services/     Application service layer
  requirements.txt

frontend/
  src/
    components/   Vue components
    views/        Page-level views
    services/     API clients
  package.json
```

## Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000` by default.
