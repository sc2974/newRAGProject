import { apiClient } from "./apiClient";

export interface AgentToolStep {
  tool: string;
  tool_input?: unknown;
  tool_output: string;
  thought?: string | null;
}

export interface AgentAskResponse {
  query: string;
  answer: string;
  generated_by: string;
  steps: AgentToolStep[];
  error?: string | null;
}

export type AgentStreamEvent =
  | {
      type: "start";
      query: string;
      generated_by: string;
    }
  | {
      type: "step";
      step: AgentToolStep;
    }
  | {
      type: "tool_start";
      run_id?: string;
      tool: string;
      tool_input?: unknown;
    }
  | {
      type: "tool_end";
      run_id?: string;
      tool: string;
      tool_output: string;
    }
  | {
      type: "answer";
      content: string;
    }
  | {
      type: "error";
      error: string;
    }
  | {
      type: "done";
    };

export async function askAgent(query: string, sessionId?: string): Promise<AgentAskResponse> {
  const response = await apiClient.post<AgentAskResponse>("/agent/ask", {
    query,
    session_id: sessionId
  });
  return response.data;
}

export async function askAgentStream(
  query: string,
  sessionId: string | undefined,
  onEvent: (event: AgentStreamEvent) => void
): Promise<void> {
  const response = await fetch("/api/agent/ask/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query, session_id: sessionId })
  });

  if (!response.ok || !response.body) {
    throw new Error("Agent stream request failed.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const rawEvent of events) {
      const line = rawEvent
        .split("\n")
        .find((item) => item.startsWith("data: "));
      if (!line) {
        continue;
      }
      try {
        onEvent(JSON.parse(line.slice(6)) as AgentStreamEvent);
      } catch (error) {
        console.warn("Failed to parse Agent stream event.", error, line);
      }
    }
  }

  if (buffer.trim()) {
    const line = buffer
      .split("\n")
    .find((item) => item.startsWith("data: "));
    if (line) {
      try {
        onEvent(JSON.parse(line.slice(6)) as AgentStreamEvent);
      } catch (error) {
        console.warn("Failed to parse Agent stream event.", error, line);
      }
    }
  }
}
