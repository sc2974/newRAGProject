<template>
  <section class="panel">
    <div class="panel-heading">
      <div>
        <p class="eyebrow">Documents</p>
        <h2>知识库文档</h2>
      </div>
      <div class="heading-actions">
        <button class="ghost-button" type="button" @click="loadDocuments">刷新</button>
        <button class="ghost-button" type="button" :disabled="reindexing" @click="reindex">
          {{ reindexing ? "重建中" : "重建索引" }}
        </button>
      </div>
    </div>

    <form class="upload-box" @submit.prevent="submit">
      <input ref="fileInput" type="file" @change="onFileChange" />
      <button type="submit" :disabled="!selectedFile || loading">
        {{ loading ? "上传中" : "上传文档" }}
      </button>
    </form>

    <form class="search-box" @submit.prevent="search">
      <input v-model="query" type="text" placeholder="搜索已入库的文档片段" />
      <button type="submit" :disabled="!query.trim() || searching">
        {{ searching ? "搜索中" : "搜索" }}
      </button>
    </form>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="notice" class="notice">{{ notice }}</p>

    <ul v-if="results.length" class="search-results">
      <li v-for="result in results" :key="result.id">
        <strong>{{ result.filename }} #{{ result.chunk_index + 1 }}</strong>
        <span v-if="result.score !== null && result.score !== undefined">
          score {{ result.score.toFixed(3) }}
        </span>
        <p>{{ result.text }}</p>
      </li>
    </ul>

    <ul class="item-list">
      <li v-for="document in documents" :key="document.id" class="list-item">
        <div>
          <strong>{{ document.filename }}</strong>
          <span>
            v{{ document.version }} · {{ formatSize(document.size_bytes) }} · {{ document.status }} ·
            {{ document.document_type }} {{ formatConfidence(document.document_type_confidence) }} ·
            {{ document.text_length }} chars · {{ document.chunk_count }} chunks
          </span>
          <small v-if="document.text_preview">{{ document.text_preview }}</small>
          <small v-if="document.parse_error">{{ document.parse_error }}</small>
          <small v-if="document.index_error">{{ document.index_error }}</small>
        </div>
        <button class="text-button" type="button" @click="remove(document.id)">删除</button>
      </li>
    </ul>

    <p v-if="!documents.length && !loading" class="empty">还没有上传文档。</p>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import {
  deleteDocument,
  listDocuments,
  reindexDocuments,
  searchDocuments,
  uploadDocument,
  type DocumentSearchResult,
  type DocumentRead
} from "../services/documentsApi";

const documents = ref<DocumentRead[]>([]);
const results = ref<DocumentSearchResult[]>([]);
const selectedFile = ref<File | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const query = ref("");
const loading = ref(false);
const searching = ref(false);
const reindexing = ref(false);
const error = ref("");
const notice = ref("");

onMounted(loadDocuments);

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  selectedFile.value = input.files?.[0] ?? null;
}

async function loadDocuments() {
  error.value = "";
  notice.value = "";
  try {
    documents.value = await listDocuments();
  } catch {
    error.value = "文档列表加载失败，请确认后端服务已经启动。";
  }
}

async function submit() {
  if (!selectedFile.value) {
    return;
  }

  loading.value = true;
  error.value = "";
  notice.value = "";

  try {
    const uploaded = await uploadDocument(selectedFile.value);
    if (uploaded.duplicate_upload) {
      documents.value = documents.value.map((document) =>
        document.id === uploaded.id ? uploaded : document
      );
      if (!documents.value.some((document) => document.id === uploaded.id)) {
        documents.value = [uploaded, ...documents.value];
      }
      notice.value = `Duplicate file skipped. Existing document: ${uploaded.filename}`;
    } else if (uploaded.replacement_upload) {
      documents.value = [
        uploaded,
        ...documents.value.filter((document) => document.id !== uploaded.replaced_document_id)
      ];
      results.value = results.value.filter(
        (result) => result.document_id !== uploaded.replaced_document_id
      );
      notice.value = `Document updated. Old index removed for: ${uploaded.filename}`;
    } else {
      documents.value = [uploaded, ...documents.value];
    }
    selectedFile.value = null;
    if (fileInput.value) {
      fileInput.value.value = "";
    }
  } catch {
    error.value = "上传失败，请稍后重试。";
  } finally {
    loading.value = false;
  }
}

async function search() {
  if (!query.value.trim()) {
    return;
  }

  searching.value = true;
  error.value = "";
  notice.value = "";

  try {
    results.value = await searchDocuments(query.value.trim());
  } catch {
    error.value = "搜索失败，请确认文档已经成功入库。";
  } finally {
    searching.value = false;
  }
}

async function reindex() {
  reindexing.value = true;
  error.value = "";
  notice.value = "";

  try {
    const result = await reindexDocuments();
    notice.value = `索引重建完成：${result.document_count} 个文档，${result.chunk_count} 个片段。`;
    await loadDocuments();
  } catch {
    error.value = "索引重建失败，请确认 Ollama embedding 模型可用。";
  } finally {
    reindexing.value = false;
  }
}

async function remove(documentId: string) {
  error.value = "";
  notice.value = "";
  try {
    await deleteDocument(documentId);
    documents.value = documents.value.filter((document) => document.id !== documentId);
    results.value = results.value.filter((result) => result.document_id !== documentId);
  } catch {
    error.value = "删除失败，请稍后重试。";
  }
}

function formatSize(size: number) {
  if (size < 1024) {
    return `${size} B`;
  }
  return `${(size / 1024).toFixed(1)} KB`;
}

function formatConfidence(confidence: number) {
  if (!confidence) {
    return "";
  }
  return `(${Math.round(confidence * 100)}%)`;
}
</script>
