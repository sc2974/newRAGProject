import { apiClient } from "./apiClient";

export interface DocumentRead {
  id: string;
  filename: string;
  content_type?: string | null;
  size_bytes: number;
  content_md5: string;
  status: "uploaded" | "parsed" | "processing" | "indexed" | "failed";
  document_type: "faq" | "legal" | "qa" | "scientific" | "code" | "tabular" | "general";
  document_type_confidence: number;
  text_length: number;
  chunk_count: number;
  text_preview: string;
  parse_error?: string | null;
  index_error?: string | null;
  duplicate_upload: boolean;
  duplicate_of?: string | null;
  replacement_upload: boolean;
  replaced_document_id?: string | null;
  version: number;
  is_active: boolean;
  deleted_at?: string | null;
  replaced_by?: string | null;
  created_at: string;
}

export interface DocumentContent {
  id: string;
  filename: string;
  text: string;
}

export interface DocumentSearchResult {
  id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  text: string;
  score?: number | null;
}

export interface DocumentReindexResult {
  document_count: number;
  chunk_count: number;
  failed_documents: string[];
}

export async function uploadDocument(file: File): Promise<DocumentRead> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post<DocumentRead>("/documents/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return response.data;
}

export async function listDocuments(): Promise<DocumentRead[]> {
  const response = await apiClient.get<DocumentRead[]>("/documents/");
  return response.data;
}

export async function getDocumentContent(documentId: string): Promise<DocumentContent> {
  const response = await apiClient.get<DocumentContent>(`/documents/${documentId}/content`);
  return response.data;
}

export async function searchDocuments(
  query: string,
  limit = 5
): Promise<DocumentSearchResult[]> {
  const response = await apiClient.post<DocumentSearchResult[]>("/documents/search", {
    query,
    limit
  });
  return response.data;
}

export async function reindexDocuments(): Promise<DocumentReindexResult> {
  const response = await apiClient.post<DocumentReindexResult>("/documents/reindex");
  return response.data;
}

export async function deleteDocument(documentId: string): Promise<void> {
  await apiClient.delete(`/documents/${documentId}`);
}
