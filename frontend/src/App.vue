<template>
  <main class="app-shell">
    <section class="hero">
      <p class="eyebrow">RAG FastAPI Demo</p>
      <div>
        <h1>智能知识库工作台</h1>
        <p>
          先把页面流程和后端 mock API 跑通。下一步再把文档解析、向量检索、
          LLM 生成逐个接进来。
        </p>
      </div>
    </section>

    <section class="dashboard">
      <div class="side-stack">
        <DocumentsPanel />
        <NotesPanel />
      </div>
      <ChatPanel />
    </section>
  </main>
</template>

<script setup lang="ts">
import ChatPanel from "./components/ChatPanel.vue";
import DocumentsPanel from "./components/DocumentsPanel.vue";
import NotesPanel from "./components/NotesPanel.vue";
</script>
