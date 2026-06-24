<template>
  <section class="panel chat-panel">
    <div class="panel-heading">
      <div>
        <p class="eyebrow">Chat</p>
        <h2>问答联调</h2>
      </div>
      <button class="ghost-button" type="button" @click="startSession">新会话</button>
    </div>

    <div class="session-strip">
      <button
        v-for="session in sessions"
        :key="session.id"
        :class="{ active: session.id === activeSession?.id }"
        type="button"
        @click="activeSession = session"
      >
        {{ session.title }}
      </button>
    </div>

    <div ref="messageListRef" class="message-list">
      <article
        v-for="chatMessage in activeSession?.messages"
        :key="chatMessage.id"
        :class="['message', chatMessage.role]"
      >
        <span>{{ chatMessage.role }}</span>
        <p>{{ chatMessage.content }}</p>
      </article>
      <p v-if="!activeSession?.messages.length" class="empty">
        输入问题后，统一由 Agent 判断是否调用 RAG、HyDE、检索工具或普通 LLM。
      </p>
    </div>

    <form class="chat-form unified-chat-form" @submit.prevent="askAgentAssistant">
      <textarea v-model="message" rows="3" placeholder="输入问题，Agent 会自动选择 LLM / RAG / 工具" />
      <div class="route-summary">
        <span>Unified Agent Entry</span>
        <strong>RAG 默认 HyDE + Auto Rerank</strong>
      </div>
      <div class="form-actions main-actions">
        <button type="submit" :disabled="!message.trim() || anyLoading">
          {{ agentLoading ? "Agent 执行中" : "智能问答" }}
        </button>
      </div>

      <details class="debug-routes">
        <summary>调试直连接口</summary>
        <div class="debug-route-grid">
          <label class="retrieval-mode-control">
            <span>Retrieval mode</span>
            <select v-model="retrievalMode">
              <option value="auto">Auto</option>
              <option value="vector">Vector</option>
              <option value="bm25">BM25</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </label>
          <label class="retrieval-mode-control">
            <span>Rerank mode</span>
            <select v-model="rerankMode">
              <option value="auto">Auto</option>
              <option value="none">None</option>
              <option value="keyword">Keyword</option>
              <option value="semantic">Semantic</option>
            </select>
          </label>
          <label class="retrieval-mode-control">
            <span>HyDE</span>
            <strong>On by default</strong>
          </label>
        </div>
        <div class="form-actions">
          <button
            class="ghost-button"
            type="button"
            :disabled="!message.trim() || anyLoading"
            @click="askKnowledge"
          >
            {{ ragLoading ? "检索中" : "直连 RAG" }}
          </button>
          <button
            class="ghost-button"
            type="button"
            :disabled="!message.trim() || anyLoading"
            @click="askPlainLLM"
          >
            {{ llmLoading ? "生成中" : "直连 LLM" }}
          </button>
          <button
            class="ghost-button"
            type="button"
            :disabled="!message.trim() || anyLoading"
            @click="send"
          >
            {{ loading ? "发送中" : "原始聊天" }}
          </button>
        </div>
      </details>
    </form>

    <article v-if="llmAnswer" class="rag-result">
      <h3>普通 LLM 回答</h3>
      <span class="generator">{{ llmAnswer.generated_by }}</span>
      <p>{{ llmAnswer.answer }}</p>
      <p v-if="llmAnswer.llm_error" class="warning">
        Ollama 调用失败：{{ llmAnswer.llm_error }}
      </p>
    </article>

    <article v-if="agentAnswer" class="rag-result">
      <h3>Agent 问答</h3>
      <span class="generator">{{ agentAnswer.generated_by }}</span>
      <p>{{ agentAnswer.answer }}</p>
      <p v-if="agentAnswer.error" class="warning">
        Agent 执行失败：{{ agentAnswer.error }}
      </p>
      <div v-if="agentAnswer.steps.length" class="agent-steps">
        <h4>工具调用步骤</h4>
        <article v-for="(step, index) in agentAnswer.steps" :key="`${step.tool}-${index}`">
          <strong>{{ index + 1 }}. {{ step.tool }}</strong>
          <span v-if="step.tool_input !== null && step.tool_input !== undefined">
            input {{ formatToolInput(step.tool_input) }}
          </span>
          <pre class="source-chunk agent-tool-output">{{ step.tool_output }}</pre>
        </article>
      </div>
    </article>

    <article v-if="ragAnswer" class="rag-result">
      <h3>RAG 检索回答</h3>
      <span class="generator">{{ ragAnswer.generated_by }}</span>
      <div :class="['retrieval-status', ragAnswer.retrieval_status]">
        <strong>{{ retrievalStatusLabel(ragAnswer.retrieval_status) }}</strong>
        <span>mode {{ ragAnswer.retrieval_mode }}</span>
        <span>rerank {{ ragAnswer.rerank_mode }}</span>
        <span>max score {{ formatScore(ragAnswer.max_score) }}</span>
        <span>threshold {{ formatScore(ragAnswer.score_threshold) }}</span>
        <span>retrieval {{ formatTime(ragAnswer.retrieval_ms) }}</span>
        <span>hyde {{ ragAnswer.hyde ? "on" : "off" }}</span>
        <span>hyde gen {{ formatTime(ragAnswer.hyde_ms) }}</span>
        <span>rerank {{ formatTime(ragAnswer.rerank_ms) }}</span>
        <span>total {{ formatTime(ragAnswer.total_retrieval_ms) }}</span>
      </div>
      <div v-if="ragAnswer.hyde_query || ragAnswer.hyde_error" class="hyde-box">
        <h4>HyDE 假设答案</h4>
        <p v-if="ragAnswer.hyde_query">{{ ragAnswer.hyde_query }}</p>
        <p v-if="ragAnswer.hyde_error" class="warning">HyDE 生成失败：{{ ragAnswer.hyde_error }}</p>
      </div>
      <p>{{ ragAnswer.answer }}</p>
      <p v-if="ragAnswer.llm_error" class="warning">
        Ollama 调用失败，已回退到检索摘要：{{ ragAnswer.llm_error }}
      </p>
      <ul v-if="ragAnswer.sources.length">
        <li v-for="source in ragAnswer.sources" :key="source.source_id">
          <strong>{{ source.citation_label }} {{ source.title }}</strong>
          <span v-if="source.score !== null && source.score !== undefined">
            score {{ source.score.toFixed(3) }}
          </span>
          <span v-if="source.rerank_score !== null && source.rerank_score !== undefined">
            rerank {{ source.rerank_score.toFixed(3) }}
          </span>
          <span v-if="source.chunk_index !== null && source.chunk_index !== undefined">
            chunk {{ source.chunk_index + 1 }}
          </span>
          <span>method {{ source.retrieval_method ?? ragAnswer.retrieval_mode }}</span>
          <pre class="source-chunk">{{ source.content }}</pre>
        </li>
      </ul>
    </article>

    <p v-if="error" class="error">{{ error }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";
import {
  createChatSession,
  getChatSession,
  listChatSessions,
  sendChatMessage,
  type ChatSession
} from "../services/chatApi";
import { askAgentStream, type AgentAskResponse, type AgentStreamEvent } from "../services/agentApi";
import { askLLM, type LLMAskResponse } from "../services/llmApi";
import { askRag, type AskResponse } from "../services/ragApi";

const sessions = ref<ChatSession[]>([]);
const activeSessionId = ref("");
const message = ref("");
type RequestedRetrievalMode = AskResponse["retrieval_mode"] | "auto";
type RequestedRerankMode = AskResponse["rerank_mode"] | "auto";

const retrievalMode = ref<RequestedRetrievalMode>("auto");
const rerankMode = ref<RequestedRerankMode>("auto");
const llmAnswer = ref<LLMAskResponse | null>(null);
const ragAnswer = ref<AskResponse | null>(null);
const agentAnswer = ref<AgentAskResponse | null>(null);
const loading = ref(false);
const llmLoading = ref(false);
const ragLoading = ref(false);
const agentLoading = ref(false);
const error = ref("");
const messageListRef = ref<HTMLElement | null>(null);
const anyLoading = computed(() => loading.value || llmLoading.value || ragLoading.value || agentLoading.value);

const activeSession = computed({
  get() {
    return sessions.value.find((session) => session.id === activeSessionId.value) ?? null;
  },
  set(session: ChatSession | null) {
    activeSessionId.value = session?.id ?? "";
  }
});

onMounted(async () => {
  await loadSessions();
  if (!sessions.value.length) {
    await startSession();
  }
});

async function loadSessions() {
  error.value = "";
  try {
    sessions.value = await listChatSessions();
    activeSessionId.value = sessions.value[0]?.id ?? "";
  } catch {
    error.value = "会话列表加载失败，请确认后端服务已经启动。";
  }
}

async function startSession() {
  error.value = "";
  try {
    const session = await createChatSession(`Demo chat ${sessions.value.length + 1}`);
    sessions.value = [session, ...sessions.value];
    activeSession.value = session;
  } catch {
    error.value = "会话创建失败，请稍后重试。";
  }
}

async function send() {
  if (!activeSession.value) {
    await startSession();
  }
  if (!activeSession.value || !message.value.trim()) {
    return;
  }

  loading.value = true;
  error.value = "";
  agentAnswer.value = null;
  ragAnswer.value = null;
  llmAnswer.value = null;

  try {
    const result = await sendChatMessage(activeSession.value.id, message.value.trim());
    replaceSession(result.session);
    await scrollMessagesToBottom();
    message.value = "";
  } catch {
    error.value = "消息发送失败，请稍后重试。";
  } finally {
    loading.value = false;
  }
}

async function askKnowledge() {
  if (!message.value.trim()) {
    return;
  }

  ragLoading.value = true;
  error.value = "";
  agentAnswer.value = null;
  llmAnswer.value = null;
  try {
    ragAnswer.value = await askRag(message.value.trim(), {
      retrievalMode: retrievalMode.value,
      rerankMode: rerankMode.value,
      candidateLimit: 20,
      hyde: true
    });
  } catch {
    error.value = "RAG 问答接口暂时不可用。";
  } finally {
    ragLoading.value = false;
  }
}

async function askPlainLLM() {
  if (!message.value.trim()) {
    return;
  }

  llmLoading.value = true;
  error.value = "";
  agentAnswer.value = null;
  ragAnswer.value = null;
  try {
    llmAnswer.value = await askLLM(message.value.trim());
  } catch {
    error.value = "普通 LLM 问答接口暂时不可用。";
  } finally {
    llmLoading.value = false;
  }
}

async function askAgentAssistant() {
  if (!message.value.trim()) {
    return;
  }
  if (!activeSession.value) {
    await startSession();
  }
  if (!activeSession.value) {
    return;
  }

  agentLoading.value = true;
  error.value = "";
  ragAnswer.value = null;
  llmAnswer.value = null;
  const query = message.value.trim();
  const sessionId = activeSession.value.id;
  agentAnswer.value = {
    query,
    answer: "",
    generated_by: "agent",
    steps: [],
    error: null
  };
  try {
    await askAgentStream(query, sessionId, handleAgentStreamEvent);
    if (agentAnswer.value?.answer) {
      const updatedSession = await getChatSession(sessionId);
      replaceSession(updatedSession);
      await scrollMessagesToBottom();
      message.value = "";
    }
  } catch {
    error.value = "Agent 问答接口暂时不可用。";
  } finally {
    agentLoading.value = false;
  }
}

function handleAgentStreamEvent(event: AgentStreamEvent) {
  if (!agentAnswer.value) {
    return;
  }
  if (event.type === "start") {
    agentAnswer.value.generated_by = event.generated_by;
    return;
  }
  if (event.type === "step") {
    agentAnswer.value.steps = [...agentAnswer.value.steps, event.step];
    return;
  }
  if (event.type === "tool_start") {
    agentAnswer.value.steps = [
      ...agentAnswer.value.steps,
      {
        tool: event.tool,
        tool_input: event.tool_input,
        tool_output: "Tool is running..."
      }
    ];
    return;
  }
  if (event.type === "tool_end") {
    const stepIndex = findRunningToolStep(event.run_id, event.tool);
    const nextStep = {
      tool: event.tool,
      tool_input: stepIndex >= 0 ? agentAnswer.value.steps[stepIndex].tool_input : undefined,
      tool_output: event.tool_output
    };
    if (stepIndex >= 0) {
      agentAnswer.value.steps = agentAnswer.value.steps.map((step, index) =>
        index === stepIndex ? nextStep : step
      );
    } else {
      agentAnswer.value.steps = [...agentAnswer.value.steps, nextStep];
    }
    return;
  }
  if (event.type === "answer") {
    agentAnswer.value.answer += event.content;
    return;
  }
  if (event.type === "error") {
    agentAnswer.value.error = event.error;
  }
}

function findRunningToolStep(_runId: string | undefined, tool: string) {
  if (!agentAnswer.value) {
    return -1;
  }
  for (let index = agentAnswer.value.steps.length - 1; index >= 0; index -= 1) {
    const step = agentAnswer.value.steps[index];
    if (step.tool === tool && step.tool_output === "Tool is running...") {
      return index;
    }
  }
  return -1;
}

function replaceSession(nextSession: ChatSession) {
  const exists = sessions.value.some((session) => session.id === nextSession.id);
  sessions.value = exists
    ? sessions.value.map((session) => (session.id === nextSession.id ? nextSession : session))
    : [nextSession, ...sessions.value];
  activeSession.value = nextSession;
}

async function scrollMessagesToBottom() {
  await nextTick();
  if (!messageListRef.value) {
    return;
  }
  messageListRef.value.scrollTop = messageListRef.value.scrollHeight;
}

function retrievalStatusLabel(status: AskResponse["retrieval_status"]) {
  if (status === "hit") {
    return "retrieval hit";
  }
  if (status === "below_threshold") {
    return "below threshold";
  }
  return "no retrieval results";
}

function formatScore(score?: number | null) {
  if (score === null || score === undefined) {
    return "-";
  }
  return score.toFixed(3);
}

function formatTime(ms?: number | null) {
  if (ms === null || ms === undefined) {
    return "-";
  }
  return `${ms.toFixed(1)} ms`;
}

function formatToolInput(input: unknown) {
  if (typeof input === "string") {
    return input;
  }
  return JSON.stringify(input);
}
</script>
