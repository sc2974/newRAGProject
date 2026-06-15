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

    <div class="message-list">
      <article
        v-for="chatMessage in activeSession?.messages"
        :key="chatMessage.id"
        :class="['message', chatMessage.role]"
      >
        <span>{{ chatMessage.role }}</span>
        <p>{{ chatMessage.content }}</p>
      </article>
      <p v-if="!activeSession?.messages.length" class="empty">
        输入问题后，可以测试聊天接口，也可以走 RAG 检索问答。
      </p>
    </div>

    <form class="chat-form" @submit.prevent="send">
      <textarea v-model="message" rows="3" placeholder="输入问题，例如：vector database persistence" />
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
      <div class="form-actions">
        <button
          class="ghost-button"
          type="button"
          :disabled="!message.trim() || ragLoading"
          @click="askKnowledge"
        >
          {{ ragLoading ? "检索中" : "RAG 问答" }}
        </button>
        <button
          class="ghost-button"
          type="button"
          :disabled="!message.trim() || llmLoading"
          @click="askPlainLLM"
        >
          {{ llmLoading ? "生成中" : "普通 LLM" }}
        </button>
        <button type="submit" :disabled="!message.trim() || loading">
          {{ loading ? "发送中" : "发送消息" }}
        </button>
      </div>
    </form>

    <article v-if="llmAnswer" class="rag-result">
      <h3>普通 LLM 回答</h3>
      <span class="generator">{{ llmAnswer.generated_by }}</span>
      <p>{{ llmAnswer.answer }}</p>
      <p v-if="llmAnswer.llm_error" class="warning">
        Ollama 调用失败：{{ llmAnswer.llm_error }}
      </p>
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
        <span>rerank {{ formatTime(ragAnswer.rerank_ms) }}</span>
        <span>total {{ formatTime(ragAnswer.total_retrieval_ms) }}</span>
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
import { computed, onMounted, ref } from "vue";
import {
  createChatSession,
  listChatSessions,
  sendChatMessage,
  type ChatSession
} from "../services/chatApi";
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
const loading = ref(false);
const llmLoading = ref(false);
const ragLoading = ref(false);
const error = ref("");

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

  try {
    const result = await sendChatMessage(activeSession.value.id, message.value.trim());
    replaceSession(result.session);
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
  try {
    ragAnswer.value = await askRag(message.value.trim(), {
      retrievalMode: retrievalMode.value,
      rerankMode: rerankMode.value,
      candidateLimit: 20
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
  try {
    llmAnswer.value = await askLLM(message.value.trim());
  } catch {
    error.value = "普通 LLM 问答接口暂时不可用。";
  } finally {
    llmLoading.value = false;
  }
}

function replaceSession(nextSession: ChatSession) {
  sessions.value = sessions.value.map((session) =>
    session.id === nextSession.id ? nextSession : session
  );
  activeSession.value = nextSession;
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
</script>
