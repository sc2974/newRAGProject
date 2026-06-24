import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.agent.tools import (
    DEFAULT_AGENT_TOOLS,
    bm25_search_tool,
    compare_retrieval_modes_tool,
    hyde_rag_search_tool,
    inspect_document_chunks_tool,
    list_documents_tool,
    rag_search_tool,
    vector_search_tool,
    what_time_is_now,
)
from app.core.config import settings
from app.rag.service import RagService
from app.schemas.agent import AgentAskRequest, AgentAskResponse, AgentToolStep
from app.schemas.rag import AskRequest
from app.services.chat_service import chat_service


class AgentService:
    def __init__(self) -> None:
        self._tools = DEFAULT_AGENT_TOOLS

    async def ask(self, request: AgentAskRequest) -> AgentAskResponse:
        try:
            direct_answer = await self._try_answer_chunk_followup(request)
            if direct_answer is not None:
                await self._save_exchange(request.session_id, request.query, direct_answer)
                return AgentAskResponse(
                    query=request.query,
                    answer=direct_answer,
                    generated_by=f"{self._model_label()}:direct-rag-chunks",
                    steps=[],
                )

            direct_tool_result = await self._try_direct_tool_route(request)
            if direct_tool_result is not None:
                answer = self._build_direct_tool_answer(direct_tool_result)
                await self._save_exchange(request.session_id, request.query, answer)
                return AgentAskResponse(
                    query=request.query,
                    answer=answer,
                    generated_by=f"{self._model_label()}:direct-tool-route",
                    steps=[
                        AgentToolStep(
                            tool=direct_tool_result["tool"],
                            tool_input=direct_tool_result["tool_input"],
                            tool_output=direct_tool_result["tool_output"],
                            thought=direct_tool_result["reason"],
                        )
                    ],
                )

            chat_history = await self._load_chat_history(request.session_id)
            executor = self._create_executor()
            result = await executor.ainvoke(
                {
                    "input": request.query,
                    "chat_history": chat_history,
                    "system_prompt": self._system_prompt(),
                }
            )
            answer = str(result.get("output", ""))
            await self._save_exchange(request.session_id, request.query, answer)
            return AgentAskResponse(
                query=request.query,
                answer=answer,
                generated_by=self._model_label(),
                steps=self._format_steps(result.get("intermediate_steps", [])),
            )
        except Exception as exc:
            return AgentAskResponse(
                query=request.query,
                answer="The agent is unavailable right now.",
                generated_by=self._model_label(),
                steps=[],
                error=str(exc),
            )

    async def stream(self, request: AgentAskRequest) -> AsyncGenerator[str, None]:
        yield self._sse(
            {
                "type": "start",
                "query": request.query,
                "generated_by": self._model_label(),
            }
        )
        try:
            direct_answer = await self._try_answer_chunk_followup(request)
            if direct_answer is not None:
                await self._save_exchange(request.session_id, request.query, direct_answer)
                for chunk in self._chunk_text(direct_answer):
                    yield self._sse({"type": "answer", "content": chunk})
                yield self._sse({"type": "done"})
                return

            direct_route = self._select_direct_tool_route(request)
            if direct_route is not None:
                yield self._sse(
                    {
                        "type": "tool_start",
                        "run_id": direct_route["run_id"],
                        "tool": direct_route["tool"],
                        "tool_input": direct_route["tool_input"],
                    }
                )
                direct_tool_result = await self._run_direct_tool_route(direct_route)
                yield self._sse(
                    {
                        "type": "tool_end",
                        "run_id": direct_route["run_id"],
                        "tool": direct_tool_result["tool"],
                        "tool_output": direct_tool_result["tool_output"],
                    }
                )
                answer = self._build_direct_tool_answer(direct_tool_result)
                await self._save_exchange(request.session_id, request.query, answer)
                for chunk in self._chunk_text(answer):
                    yield self._sse({"type": "answer", "content": chunk})
                yield self._sse({"type": "done"})
                return

            chat_history = await self._load_chat_history(request.session_id)
            executor = self._create_executor()
            agent_input = {
                "input": request.query,
                "chat_history": chat_history,
                "system_prompt": self._system_prompt(),
            }
            final_result: dict[str, Any] | None = None
            async for event in executor.astream_events(agent_input, version="v2"):
                event_type = event.get("event")
                event_name = event.get("name")
                run_id = event.get("run_id")
                data = event.get("data") or {}

                if event_type == "on_tool_start":
                    yield self._sse(
                        {
                            "type": "tool_start",
                            "run_id": run_id,
                            "tool": event_name,
                            "tool_input": data.get("input"),
                        }
                    )
                    continue

                if event_type == "on_tool_end":
                    yield self._sse(
                        {
                            "type": "tool_end",
                            "run_id": run_id,
                            "tool": event_name,
                            "tool_output": self._stringify_tool_output(data.get("output")),
                        }
                    )
                    continue

                if event_type == "on_chain_end" and event_name == "AgentExecutor":
                    output = data.get("output")
                    if isinstance(output, dict):
                        final_result = output

            answer = str(final_result.get("output", "") if final_result else "")
            if not answer:
                raise ValueError("Agent returned an empty response.")

            await self._save_exchange(request.session_id, request.query, answer)

            for chunk in self._chunk_text(answer):
                yield self._sse({"type": "answer", "content": chunk})
            yield self._sse({"type": "done"})
        except Exception as exc:
            yield self._sse({"type": "error", "error": str(exc)})
            yield self._sse({"type": "done"})

    def _create_executor(self) -> AgentExecutor:
        chat_model = self._create_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(chat_model, self._tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=False,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=settings.agent_max_iterations,
        )

    def _create_chat_model(self):
        provider = settings.llm_provider.strip().lower()
        if provider == "dashscope":
            return ChatTongyi(
                model=settings.dashscope_model,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
                streaming=False,
                disable_streaming=True,
                temperature=0.1,
            )
        raise ValueError(
            f"Agent currently supports LLM_PROVIDER='dashscope'. "
            f"Current value: '{settings.llm_provider}'."
        )

    def _model_label(self) -> str:
        provider = settings.llm_provider.strip().lower()
        model = settings.dashscope_model if provider == "dashscope" else settings.ollama_model
        return f"agent:{provider}:{model}"

    def _system_prompt(self) -> str:
        return (
            "You are a tool-using AI assistant for a RAG knowledge-base demo. "
            "You can answer general questions directly, but prefer tools whenever the "
            "question asks about uploaded documents, retrieval behavior, chunks, citations, "
            "or knowledge-base content. Use list_documents_tool to inspect available files. "
            "Use inspect_document_chunks_tool when the user asks how a document was split. "
            "Use vector_search_tool for semantic search diagnostics, bm25_search_tool for "
            "keyword matching diagnostics, compare_retrieval_modes_tool when the user asks "
            "to compare vector, BM25, and hybrid retrieval, rag_search_tool when the "
            "user wants a normal grounded final answer from the knowledge base; this "
            "normal RAG path uses HyDE by default. Use "
            "hyde_rag_search_tool when the user explicitly asks for HyDE, enhanced semantic "
            "retrieval, or a vague query that may need query expansion. Use get_user_info_tools "
            "only when the user provides a JWT token and asks about user identity. Use chat "
            "history to resolve follow-up questions and references like continue, previous "
            "question, or it. If the user asks for current time or date, call what_time_is_now. "
            "After using tools, summarize what happened clearly, include important ids, and "
            "cite source labels returned by RAG when available."
        )

    async def _try_direct_tool_route(
        self,
        request: AgentAskRequest,
    ) -> dict[str, Any] | None:
        route = self._select_direct_tool_route(request)
        if route is None:
            return None
        return await self._run_direct_tool_route(route)

    def _select_direct_tool_route(self, request: AgentAskRequest) -> dict[str, Any] | None:
        query = request.query.strip()
        normalized = query.lower()
        limit = self._extract_limit(normalized)
        document_id = self._extract_document_id(query)

        if self._asks_current_time(normalized):
            return self._route("what_time_is_now", what_time_is_now, {}, "Matched current time/date intent.")

        if self._asks_list_documents(normalized):
            return self._route(
                "list_documents_tool",
                list_documents_tool,
                {},
                "Matched knowledge-base document inventory intent.",
            )

        if self._asks_document_chunks(normalized):
            if document_id:
                return self._route(
                    "inspect_document_chunks_tool",
                    inspect_document_chunks_tool,
                    {"document_id": document_id, "limit": limit},
                    "Matched document chunk inspection intent with a document_id.",
                )
            return self._route(
                "list_documents_tool",
                list_documents_tool,
                {},
                "Matched chunk inspection intent but no document_id was provided; listing documents first.",
            )

        if self._asks_retrieval_comparison(normalized):
            return self._route(
                "compare_retrieval_modes_tool",
                compare_retrieval_modes_tool,
                {"query": query, "limit": limit},
                "Matched retrieval comparison intent.",
            )

        if self._asks_bm25_search(normalized):
            return self._route(
                "bm25_search_tool",
                bm25_search_tool,
                {"query": query, "limit": limit},
                "Matched BM25-only retrieval intent.",
            )

        if self._asks_vector_search(normalized):
            return self._route(
                "vector_search_tool",
                vector_search_tool,
                {"query": query, "limit": limit},
                "Matched vector-only retrieval intent.",
            )

        if self._asks_hyde_rag(normalized):
            return self._route(
                "hyde_rag_search_tool",
                hyde_rag_search_tool,
                {"query": query},
                "Matched explicit HyDE RAG intent.",
            )

        if self._asks_rag_report(normalized):
            if self._asks_hyde_rag(normalized):
                return self._route(
                    "hyde_rag_search_tool",
                    hyde_rag_search_tool,
                    {"query": query},
                    "Matched explicit HyDE RAG knowledge-base report intent.",
                )
            return self._route(
                "rag_search_tool",
                rag_search_tool,
                {"query": query},
                "Matched explicit RAG knowledge-base report intent.",
            )

        return None

    def _route(self, tool_name: str, tool_obj, tool_input: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "run_id": f"direct-{tool_name}",
            "tool": tool_name,
            "tool_obj": tool_obj,
            "tool_input": tool_input,
            "reason": reason,
        }

    async def _run_direct_tool_route(self, route: dict[str, Any]) -> dict[str, Any]:
        tool_output = await route["tool_obj"].ainvoke(route["tool_input"])
        return {
            "run_id": route["run_id"],
            "tool": route["tool"],
            "tool_input": route["tool_input"],
            "tool_output": self._stringify_tool_output(tool_output),
            "reason": route["reason"],
        }

    def _build_direct_tool_answer(self, result: dict[str, Any]) -> str:
        return (
            f"I used {result['tool']} directly because this request matched a stable tool route.\n"
            f"Reason: {result['reason']}\n\n"
            f"{result['tool_output']}"
        )

    def _extract_limit(self, normalized_query: str) -> int:
        match = re.search(r"(?:top|limit|前|显示|返回)\s*(\d+)", normalized_query)
        if not match:
            return 5
        return max(1, min(int(match.group(1)), 10))

    def _extract_document_id(self, query: str) -> str | None:
        match = re.search(r"\bdoc_[a-zA-Z0-9]+\b", query)
        return match.group(0) if match else None

    def _asks_current_time(self, normalized_query: str) -> bool:
        return any(term in normalized_query for term in ("what time", "current time", "\u51e0\u70b9", "\u73b0\u5728\u65f6\u95f4"))

    def _asks_list_documents(self, normalized_query: str) -> bool:
        document_terms = ("document", "documents", "\u6587\u6863", "\u6587\u4ef6", "\u77e5\u8bc6\u5e93")
        list_terms = ("list", "show", "inventory", "\u5217\u51fa", "\u6709\u54ea\u4e9b", "\u5f53\u524d")
        return any(term in normalized_query for term in document_terms) and any(
            term in normalized_query for term in list_terms
        )

    def _asks_document_chunks(self, normalized_query: str) -> bool:
        chunk_terms = ("chunk", "chunks", "\u5207\u7247", "\u5206\u7247", "\u7247\u6bb5")
        inspect_terms = ("inspect", "show", "split", "\u67e5\u770b", "\u5c55\u793a", "\u600e\u4e48\u5207", "\u5982\u4f55\u5207")
        return any(term in normalized_query for term in chunk_terms) and any(
            term in normalized_query for term in inspect_terms
        )

    def _asks_retrieval_comparison(self, normalized_query: str) -> bool:
        compare_terms = ("compare", "comparison", "\u5bf9\u6bd4", "\u6bd4\u8f83")
        retrieval_terms = ("retrieval", "vector", "bm25", "hybrid", "\u68c0\u7d22", "\u5411\u91cf", "\u6df7\u5408")
        tri_terms = ("\u4e09\u79cd", "\u4e09\u4e2a", "three")
        return any(term in normalized_query for term in compare_terms) and (
            any(term in normalized_query for term in retrieval_terms)
            or any(term in normalized_query for term in tri_terms)
        )

    def _asks_bm25_search(self, normalized_query: str) -> bool:
        return "bm25" in normalized_query and not self._asks_retrieval_comparison(normalized_query)

    def _asks_vector_search(self, normalized_query: str) -> bool:
        vector_terms = ("vector", "\u5411\u91cf")
        return any(term in normalized_query for term in vector_terms) and not self._asks_retrieval_comparison(normalized_query)

    def _asks_rag_report(self, normalized_query: str) -> bool:
        rag_terms = ("rag", "\u77e5\u8bc6\u5e93")
        report_terms = ("report", "source", "sources", "citation", "\u5f15\u7528", "\u6765\u6e90", "\u62a5\u544a")
        return any(term in normalized_query for term in rag_terms) and any(
            term in normalized_query for term in report_terms
        )

    def _asks_hyde_rag(self, normalized_query: str) -> bool:
        hyde_terms = ("hyde", "hypothetical", "\u5047\u8bbe\u7b54\u6848", "\u5047\u8bbe\u6587\u6863")
        enhanced_terms = ("enhanced", "semantic expansion", "\u589e\u5f3a\u68c0\u7d22", "\u8bed\u4e49\u6269\u5c55")
        return any(term in normalized_query for term in hyde_terms + enhanced_terms)

    async def _try_answer_chunk_followup(self, request: AgentAskRequest) -> str | None:
        if not self._is_chunk_followup(request.query):
            return None

        previous_query = await self._previous_user_query(request.session_id)
        if previous_query is None:
            return (
                "????????????????????????? RAG ???"
                "??????? chunk ?????"
            )

        rag_response = await RagService().ask(
            AskRequest(
                query=previous_query,
                retrieval_mode="auto",
                rerank_mode="auto",
                limit=4,
                candidate_limit=20,
                hyde=True,
            )
        )
        if not rag_response.sources:
            return (
                f"Previous question: {previous_query}\n\n"
                f"No matched chunks were returned after rerunning retrieval. "
                f"Status: {rag_response.retrieval_status}."
            )

        lines = [
            "Matched chunks for the previous question:",
            f"- Previous question: {previous_query}",
            f"- Retrieval mode: {rag_response.retrieval_mode}",
            f"- Rerank mode: {rag_response.rerank_mode}",
            f"- HyDE enabled: {rag_response.hyde}",
            f"- Matched chunks: {len(rag_response.sources)}",
            "",
        ]
        for source in rag_response.sources:
            score = "-" if source.score is None else f"{source.score:.4f}"
            rerank_score = "-" if source.rerank_score is None else f"{source.rerank_score:.4f}"
            lines.extend(
                [
                    f"{source.citation_label} {source.title}",
                    f"  document_id: {source.document_id}",
                    f"  chunk_index: {source.chunk_index}",
                    f"  retrieval_method: {source.retrieval_method}",
                    f"  score: {score}",
                    f"  rerank_score: {rerank_score}",
                    f"  content: {source.content}",
                    "",
                ]
            )
        return "\n".join(lines)

    def _is_chunk_followup(self, query: str) -> bool:
        normalized = query.lower()
        direct_patterns = (
            "\u521a\u624d\u90a3\u4e2a\u95ee\u9898\u7684\u547d\u4e2d chunk \u662f\u54ea\u4e9b",
            "\u521a\u624d\u90a3\u4e2a\u95ee\u9898\u7684\u547d\u4e2dchunk\u662f\u54ea\u4e9b",
        )
        if any(pattern in normalized for pattern in direct_patterns):
            return True

        chunk_terms = ("chunk",)
        hit_terms = ("hit", "matched", "retrieved")
        previous_terms = ("previous", "last question", "previous question")
        return (
            any(term in normalized for term in chunk_terms)
            and any(term in normalized for term in hit_terms)
            and any(term in normalized for term in previous_terms)
        )

    async def _previous_user_query(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        session = await chat_service.get_session(session_id)
        if session is None:
            return None

        current_followup_skipped = False
        for message in reversed(session.messages):
            if message.role != "user":
                continue
            if not current_followup_skipped and self._is_chunk_followup(message.content):
                current_followup_skipped = True
                continue
            return message.content
        return None

    async def _load_chat_history(self, session_id: str | None) -> list[BaseMessage]:
        if not session_id:
            return []
        session = await chat_service.get_session(session_id)
        if session is None:
            return []

        history: list[BaseMessage] = []
        for message in session.messages[-12:]:
            if message.role == "user":
                history.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                history.append(AIMessage(content=message.content))
        return history

    async def _save_exchange(
        self,
        session_id: str | None,
        query: str,
        answer: str,
    ) -> None:
        if not session_id:
            return
        await chat_service.add_exchange(session_id, query, answer)

    def _format_steps(self, intermediate_steps: list[Any]) -> list[AgentToolStep]:
        steps = []
        for action, observation in intermediate_steps:
            steps.append(
                AgentToolStep(
                    tool=getattr(action, "tool", "unknown"),
                    tool_input=getattr(action, "tool_input", None),
                    tool_output=str(observation),
                    thought=getattr(action, "log", None),
                )
            )
        return steps

    def _stringify_tool_output(self, output: Any) -> str:
        if output is None:
            return ""
        content = getattr(output, "content", None)
        if content is not None:
            return str(content)
        return str(output)

    def _chunk_text(self, text: str, chunk_size: int = 18) -> list[str]:
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    def _sse(self, payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
