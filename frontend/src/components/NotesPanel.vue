<template>
  <section class="panel notes-panel">
    <div class="panel-heading">
      <div>
        <p class="eyebrow">Notes</p>
        <h2>笔记草稿</h2>
      </div>
      <button class="ghost-button" type="button" @click="loadNotes">刷新</button>
    </div>

    <form class="stack-form" @submit.prevent="submit">
      <input v-model="title" type="text" placeholder="标题" />
      <textarea v-model="content" rows="4" placeholder="写一点笔记内容" />
      <input v-model="tagsText" type="text" placeholder="标签，用逗号分隔" />
      <button type="submit" :disabled="!title.trim() || loading">
        {{ loading ? "保存中" : "保存笔记" }}
      </button>
    </form>

    <p v-if="error" class="error">{{ error }}</p>

    <ul class="item-list">
      <li v-for="note in notes" :key="note.id" class="list-item note-item">
        <div>
          <strong>{{ note.title }}</strong>
          <span>{{ note.content || "暂无内容" }}</span>
          <small v-if="note.tags.length">{{ note.tags.join(" / ") }}</small>
        </div>
        <button class="text-button" type="button" @click="remove(note.id)">删除</button>
      </li>
    </ul>

    <p v-if="!notes.length && !loading" class="empty">还没有笔记。</p>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { createNote, deleteNote, listNotes, type NoteRead } from "../services/notesApi";

const notes = ref<NoteRead[]>([]);
const title = ref("");
const content = ref("");
const tagsText = ref("");
const loading = ref(false);
const error = ref("");

onMounted(loadNotes);

async function loadNotes() {
  error.value = "";
  try {
    notes.value = await listNotes();
  } catch {
    error.value = "笔记列表加载失败，请确认后端服务已经启动。";
  }
}

async function submit() {
  loading.value = true;
  error.value = "";

  try {
    const note = await createNote({
      title: title.value.trim(),
      content: content.value.trim(),
      tags: parseTags(tagsText.value)
    });
    notes.value = [note, ...notes.value];
    title.value = "";
    content.value = "";
    tagsText.value = "";
  } catch {
    error.value = "笔记保存失败，请稍后重试。";
  } finally {
    loading.value = false;
  }
}

async function remove(noteId: string) {
  error.value = "";
  try {
    await deleteNote(noteId);
    notes.value = notes.value.filter((note) => note.id !== noteId);
  } catch {
    error.value = "删除失败，请稍后重试。";
  }
}

function parseTags(value: string) {
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
}
</script>
