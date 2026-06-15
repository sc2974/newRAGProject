import { apiClient } from "./apiClient";

export interface LLMAskResponse {
  query: string;
  answer: string;
  generated_by: string;
  llm_error?: string | null;
}

export async function askLLM(query: string): Promise<LLMAskResponse> {
  const response = await apiClient.post<LLMAskResponse>("/llm/ask", { query });
  return response.data;
}
