import { apiClient } from "./apiClient";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

export interface ChatReply {
  session: ChatSession;
  reply: ChatMessage;
}

export async function createChatSession(title = "Demo chat"): Promise<ChatSession> {
  const response = await apiClient.post<ChatSession>("/chat/sessions", { title });
  return response.data;
}

export async function listChatSessions(): Promise<ChatSession[]> {
  const response = await apiClient.get<ChatSession[]>("/chat/sessions");
  return response.data;
}

export async function getChatSession(sessionId: string): Promise<ChatSession> {
  const response = await apiClient.get<ChatSession>(`/chat/sessions/${sessionId}`);
  return response.data;
}

export async function sendChatMessage(
  sessionId: string,
  content: string
): Promise<ChatReply> {
  const response = await apiClient.post<ChatReply>(
    `/chat/sessions/${sessionId}/messages`,
    { content }
  );
  return response.data;
}
