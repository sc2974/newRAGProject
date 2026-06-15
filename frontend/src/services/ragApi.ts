import { apiClient } from "./apiClient";

export interface SourceDocument {
  source_id: string;
  citation_label: string;
  title: string;
  filename?: string | null;
  content: string;
  content_preview: string;
  score?: number | null;
  rerank_score?: number | null;
  retrieval_method?: string;
  document_id?: string | null;
  chunk_index?: number | null;
}

export interface AskResponse {
  query: string;
  answer: string;
  sources: SourceDocument[];
  used_retrieval: boolean;
  retrieval_status: "hit" | "no_results" | "below_threshold";
  max_score?: number | null;
  score_threshold?: number | null;
  retrieval_mode: "vector" | "bm25" | "hybrid";
  rerank: boolean;
  rerank_mode: "none" | "keyword" | "semantic";
  retrieval_ms?: number | null;
  rerank_ms?: number | null;
  total_retrieval_ms?: number | null;
  generated_by: string;
  llm_error?: string | null;
}

export async function askRag(
  query: string,
  options?: {
    scoreThreshold?: number;
    retrievalMode?: "vector" | "bm25" | "hybrid" | "auto";
    candidateLimit?: number;
    rerank?: boolean;
    rerankMode?: "none" | "keyword" | "semantic" | "auto";
  }
): Promise<AskResponse> {
  const response = await apiClient.post<AskResponse>("/rag/ask", {
    query,
    ...(options?.scoreThreshold === undefined
      ? {}
      : { score_threshold: options.scoreThreshold }),
    ...(options?.retrievalMode === undefined
      ? {}
      : { retrieval_mode: options.retrievalMode }),
    ...(options?.candidateLimit === undefined
      ? {}
      : { candidate_limit: options.candidateLimit }),
    ...(options?.rerank === undefined ? {} : { rerank: options.rerank }),
    ...(options?.rerankMode === undefined ? {} : { rerank_mode: options.rerankMode })
  });
  return response.data;
}
