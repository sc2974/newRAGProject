<template>
  <section class="chat-panel">
    <header>
      <span class="status-dot"></span>
      <div>
        <h2>问答接口预览</h2>
        <p>调用后端 <code>/api/rag/ask</code> 占位接口。</p>
      </div>
    </header>

    <form @submit.prevent="submit">
      <textarea
        v-model="query"
        rows="4"
        placeholder="输入一个问题，例如：这个 Demo 接下来要实现什么？"
      />
      <button type="submit" :disabled="loading || !query.trim()">
        {{ loading ? "请求中..." : "发送问题" }}
      </button>
    </form>

    <article v-if="answer" class="answer">
      <h3>回答</h3>
      <div :class="['retrieval-status', answer.retrieval_status]">
        <strong>{{ retrievalStatusLabel(answer.retrieval_status) }}</strong>
        <span>max score {{ formatScore(answer.max_score) }}</span>
        <span>threshold {{ formatScore(answer.score_threshold) }}</span>
      </div>
      <p>{{ answer.answer }}</p>

      <div v-if="answer.sources.length" class="sources">
        <h3>来源</h3>
        <ul>
          <li v-for="source in answer.sources" :key="source.source_id">
            <strong>{{ source.citation_label }} {{ source.title }}</strong>
            <pre class="source-chunk">{{ source.content }}</pre>
          </li>
        </ul>
      </div>
    </article>

    <p v-if="error" class="error">{{ error }}</p>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { askRag, type AskResponse } from "../services/ragApi";

const query = ref("");
const answer = ref<AskResponse | null>(null);
const loading = ref(false);
const error = ref("");

async function submit() {
  error.value = "";
  loading.value = true;

  try {
    answer.value = await askRag(query.value);
  } catch {
    error.value = "后端接口暂时不可用，请确认 FastAPI 服务已经启动。";
  } finally {
    loading.value = false;
  }
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
</script>
