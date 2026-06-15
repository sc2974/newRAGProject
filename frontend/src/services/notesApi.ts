import { apiClient } from "./apiClient";

export interface NoteRead {
  id: string;
  title: string;
  content: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface NoteInput {
  title: string;
  content: string;
  tags: string[];
}

export async function createNote(input: NoteInput): Promise<NoteRead> {
  const response = await apiClient.post<NoteRead>("/notes/", input);
  return response.data;
}

export async function listNotes(): Promise<NoteRead[]> {
  const response = await apiClient.get<NoteRead[]>("/notes/");
  return response.data;
}

export async function updateNote(
  noteId: string,
  input: Partial<NoteInput>
): Promise<NoteRead> {
  const response = await apiClient.put<NoteRead>(`/notes/${noteId}`, input);
  return response.data;
}

export async function deleteNote(noteId: string): Promise<void> {
  await apiClient.delete(`/notes/${noteId}`);
}
