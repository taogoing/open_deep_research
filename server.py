"""Standalone FastAPI server for the checkpointed deep-research graph."""

import asyncio
import base64
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.deep_researcher import _model_runtime_config, configurable_model, deep_researcher
from open_deep_research.utils import get_api_key_for_model, get_tavily_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("open_deep_research.server")

app = FastAPI(title="Open Deep Research API")
threads: dict[str, dict[str, Any]] = {}
knowledge_bases: dict[str, dict[str, Any]] = {}
KNOWLEDGE_DIR = Path(os.environ.get("ODR_DATA_DIR", "./data")) / "knowledge"


def _load_knowledge_bases() -> None:
    """Restore normalized LangChain documents from the persistent store."""
    if not KNOWLEDGE_DIR.exists():
        return
    for path in KNOWLEDGE_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            documents = [
                Document(page_content=item["page_content"], metadata=item.get("metadata") or {})
                for item in payload.get("documents", [])
            ]
            if documents:
                knowledge_bases[payload["id"]] = {**payload, "documents": documents}
        except Exception as exc:
            logger.warning("Skipping invalid knowledge store %s: %s", path.name, exc)


def _persist_knowledge_base(item: dict[str, Any]) -> None:
    """Persist normalized chunks so container restarts do not invalidate UI references."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": item["id"],
        "name": item["name"],
        "documents": [
            {"page_content": document.page_content, "metadata": document.metadata}
            for document in item["documents"]
        ],
    }
    (KNOWLEDGE_DIR / f"{item['id']}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


_load_knowledge_bases()


def _fallback_title(query: str) -> str:
    """Create a stable short title when a title model is unavailable."""
    compact = re.sub(r"\s+", " ", query).strip()
    compact = re.sub(r"^(?:请?帮我|麻烦|我想|请你|请)(?:调研|研究|分析|了解)?", "", compact).strip(" ，。！？")
    return (compact[:24] + "…") if len(compact) > 24 else (compact or "新研究")


def _extract_knowledge(name: str, encoded: str) -> list[Document]:
    """Parse an upload into LangChain documents with source/page metadata."""
    raw = encoded.split(",", 1)[1] if "," in encoded else encoded
    payload = base64.b64decode(raw)
    if name.lower().endswith(".pdf"):
        import fitz
        document = fitz.open(stream=payload, filetype="pdf")
        return [
            Document(page_content=page.get_text(), metadata={"source": name, "page": index + 1})
            for index, page in enumerate(document)
            if page.get_text().strip()
        ]
    text = payload.decode("utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": name})]


def _knowledge_context(ids: list[str], query: str, limit: int = 14000) -> str:
    """Retrieve local chunks through LangChain's BM25 retriever."""
    documents: list[Document] = []
    for item_id in ids:
        item = knowledge_bases.get(item_id)
        if item:
            documents.extend(item["documents"])
    if not documents:
        return ""
    retriever = BM25Retriever.from_documents(documents, k=min(8, len(documents)))
    results = retriever.invoke(query)
    selected, length = [], 0
    for document in results:
        if length + len(document.page_content) > limit:
            break
        source = document.metadata.get("source", "本地资料")
        page = f"，第 {document.metadata['page']} 页" if document.metadata.get("page") else ""
        selected.append(f"[知识库：{source}{page}]\n{document.page_content}")
        length += len(document.page_content)
    return "\n\n".join(selected)


async def _generate_title(query: str, configurable: dict[str, Any]) -> str:
    """Ask the selected lightweight model for a compact conversation title."""
    fallback = _fallback_title(query)
    try:
        parsed = Configuration.from_runnable_config({"configurable": configurable})
        if not parsed.research_model.lower().startswith("ollama:") and not get_api_key_for_model(
            parsed.research_model, {"configurable": configurable}
        ):
            return fallback
        settings = _model_runtime_config(parsed.research_model, 80, parsed, {"configurable": configurable})
        result = await asyncio.wait_for(
            configurable_model.with_config(settings).ainvoke([
                HumanMessage(content=f"将下面的研究问题改写成一个不超过16个汉字的会话标题。只输出标题，不加引号或标点。\n\n{query}")
            ]), timeout=8,
        )
        title = re.sub(r"[\r\n\"'《》]", "", str(result.content)).strip(" 。！？:：")
        return title[:24] or fallback
    except Exception as exc:
        logger.warning("Title generation unavailable; using local fallback: %s", exc)
        return fallback


def _message_to_wire(message: BaseMessage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    role = "assistant" if message.type in ("ai", "assistant") else "user"
    return {"role": role, "type": message.type, "content": message.content}


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return _message_to_wire(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return _json_safe(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(_json_safe(data), ensure_ascii=False)}\n\n"


def _validate_runtime_config(configurable: dict[str, Any]) -> Configuration:
    config = Configuration.from_runnable_config({"configurable": configurable})
    runnable_config = {"configurable": configurable}
    model_fields = (
        config.research_model,
        config.compression_model,
        config.final_report_model,
        config.summarization_model,
    )
    missing_models = sorted({
        model for model in model_fields
        if model.split(":", 1)[0] in {"openai", "anthropic", "google", "deepseek"}
        and not get_api_key_for_model(model, runnable_config)
    })
    if missing_models:
        provider = missing_models[0].split(":", 1)[0].upper()
        raise ValueError(
            f"模型 {', '.join(missing_models)} 缺少凭据，请在 .env 中设置 "
            f"{provider}_API_KEY 后重启服务。"
        )

    sources = set(config.search_apis)
    if SearchAPI.TAVILY in sources and not get_tavily_api_key(runnable_config):
        raise ValueError("Tavily 搜索缺少凭据，请设置 TAVILY_API_KEY 后重启服务。")
    if SearchAPI.OPENAI in sources and not config.research_model.startswith("openai:"):
        raise ValueError("OpenAI 原生搜索只能搭配 OpenAI 研究模型。")
    if SearchAPI.ANTHROPIC in sources and not config.research_model.startswith("anthropic:"):
        raise ValueError("Anthropic 原生搜索只能搭配 Anthropic 研究模型。")
    return config


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "missing credentials" in lowered and "openai" in lowered:
        return "OpenAI 模型缺少凭据。请改用 DeepSeek，或设置 OPENAI_API_KEY。"
    if "connection refused" in lowered and "11434" in lowered:
        return "无法连接 Ollama。请确认 Ollama 已启动，并检查服务地址。"
    if "authentication" in lowered or "api key" in lowered:
        return f"模型或搜索服务认证失败：{message}"
    return message


def _extract_streaming_text(chunk: Any) -> str | None:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                return block["text"]
    return None


STEP_META = {
    "clarify_with_user": ("scope", "理解研究问题"),
    "request_clarification": ("scope", "等待补充研究范围"),
    "write_research_plan": ("plan", "生成研究方案"),
    "review_research_plan": ("plan", "等待确认研究方案"),
    "research_supervisor": ("research", "执行深度研究"),
    "final_report_generation": ("report", "生成最终报告"),
}


def _match_step(name: str) -> str | None:
    if name in STEP_META:
        return name
    return next((step for step in STEP_META if name.startswith(step)), None)


def _progress(step: str, status: str) -> dict[str, Any]:
    phase, title = STEP_META[step]
    return {
        "type": "phase.updated",
        "phase": phase,
        "step": step,
        "status": status,
        "title": title,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _extract_interrupt(snapshot: Any) -> dict[str, Any] | None:
    for task in getattr(snapshot, "tasks", ()):
        for item in getattr(task, "interrupts", ()):
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return _json_safe(value)
    return None


async def _research_event_stream(
    thread_id: str,
    *,
    input_data: dict[str, Any] | None = None,
    resume_data: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """Run or resume one graph thread and expose a stable product event stream."""
    thread = threads.setdefault(thread_id, {
        "state": {}, "configurable": {}, "pending": None,
        "status": "idle", "activities": [],
    })
    if input_data is not None:
        configurable = (
            input_data.get("configurable")
            or input_data.get("config", {}).get("configurable", {})
        )
        thread["configurable"] = configurable
        new_messages = input_data.get("messages") or input_data.get("input", {}).get("messages", [])
        knowledge_ids = configurable.get("knowledge_base_ids") or []
        user_query = new_messages[-1].get("content", "") if new_messages else ""
        local_context = _knowledge_context(knowledge_ids, user_query) if knowledge_ids else ""
        graph_input: Any = {
            "messages": [HumanMessage(content=(
                item.get("content", "") + (
                    "\n\n以下是与问题相关的本地知识库片段。请将其作为内部资料使用，并在报告中标注对应知识库文件名：\n" + local_context
                    if local_context and item is new_messages[-1] else ""
                )
            )) for item in new_messages]
        }
    else:
        configurable = thread.get("configurable", {})
        graph_input = Command(resume=resume_data or {})

    run_config = {"configurable": {**configurable, "thread_id": thread_id}}
    active_step: str | None = None
    thread["status"] = "running"
    thread["pending"] = None

    try:
        _validate_runtime_config(configurable)
        yield _sse("run", {"status": "running", "thread_id": thread_id})

        async for event in deep_researcher.astream_events(graph_input, run_config, version="v2"):
            if thread.get("stop_requested"):
                thread["status"] = "stopped"
                yield _sse("end", {"status": "stopped"})
                return
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {}) or {}
            metadata = event.get("metadata", {}) or {}

            if kind == "on_chain_start":
                matched = _match_step(name)
                if matched:
                    active_step = matched
                    yield _sse("progress", _progress(matched, "running"))

            if kind == "on_chain_end":
                matched = _match_step(name)
                if matched and matched not in {"request_clarification", "review_research_plan"}:
                    yield _sse("progress", _progress(matched, "completed"))
                    if active_step == matched:
                        active_step = None

            if kind == "on_custom_event" and name == "research_activity":
                activity = _json_safe(data)
                if isinstance(activity, dict) and "data" in activity and len(activity) == 1:
                    activity = activity["data"]
                activity = {**activity, "timestamp": datetime.now(timezone.utc).isoformat()}
                thread["activities"].append(activity)
                thread["activities"] = thread["activities"][-200:]
                yield _sse("activity", activity)

            stream_node = _match_step(str(metadata.get("langgraph_node", "")))
            if kind == "on_chat_model_stream" and stream_node == "final_report_generation":
                text = _extract_streaming_text(data.get("chunk"))
                if text:
                    yield _sse("report/partial", {"content": text})

        snapshot = await deep_researcher.aget_state(run_config)
        state = _json_safe(snapshot.values or {})
        thread["state"] = state
        pending = _extract_interrupt(snapshot)

        if pending:
            thread["pending"] = pending
            thread["status"] = "waiting"
            yield _sse("interaction", pending)
            yield _sse("run", {"status": "waiting", "interaction": pending.get("type")})
            return

        thread["status"] = "completed"
        yield _sse("state", state)
        yield _sse("end", {"status": "completed"})

    except asyncio.CancelledError:
        thread["status"] = "stopped"
        raise
    except Exception as exc:
        logger.exception("Research stream error")
        thread["status"] = "failed"
        if active_step:
            yield _sse("progress", _progress(active_step, "failed"))
        yield _sse("error", {"message": _friendly_error(exc)})


def _stream_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/threads")
async def create_thread(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    thread_id = str(uuid.uuid4())
    threads[thread_id] = {
        "state": {}, "configurable": {}, "pending": None,
        "status": "idle", "activities": [], "stop_requested": False,
    }
    title = await _generate_title(payload.get("query", ""), payload.get("configurable") or {})
    return {"thread_id": thread_id, "title": title}


@app.post("/threads/{thread_id}/runs/stream")
async def run_stream(thread_id: str, request: Request):
    if thread_id in threads:
        threads[thread_id]["stop_requested"] = False
    return _stream_response(_research_event_stream(thread_id, input_data=await request.json()))


@app.post("/threads/{thread_id}/runs/stop")
async def stop_run(thread_id: str):
    thread = threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="研究任务不存在")
    thread["stop_requested"] = True
    thread["status"] = "stopped"
    thread["pending"] = None
    return {"status": "stopped"}


@app.post("/knowledge")
async def add_knowledge(request: Request):
    payload = await request.json()
    name = str(payload.get("name") or "未命名资料")
    try:
        source_documents = _extract_knowledge(name, str(payload.get("content") or ""))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析知识文件：{exc}") from exc
    if not source_documents:
        raise HTTPException(status_code=400, detail="知识文件中没有可检索文本")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=180)
    documents = splitter.split_documents(source_documents)
    item_id = str(uuid.uuid4())
    for document in documents:
        document.metadata["knowledge_base_id"] = item_id
    knowledge_bases[item_id] = {"id": item_id, "name": name, "documents": documents}
    _persist_knowledge_base(knowledge_bases[item_id])
    return {"id": item_id, "name": name, "chunks": len(documents)}


@app.get("/knowledge")
async def list_knowledge():
    return {
        "items": [
            {"id": item["id"], "name": item["name"], "chunks": len(item["documents"])}
            for item in knowledge_bases.values()
        ]
    }


@app.delete("/knowledge/{item_id}")
async def delete_knowledge(item_id: str):
    knowledge_bases.pop(item_id, None)
    (KNOWLEDGE_DIR / f"{item_id}.json").unlink(missing_ok=True)
    return {"status": "deleted"}


@app.post("/threads/{thread_id}/runs/resume")
async def resume_stream(thread_id: str, request: Request):
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="研究任务不存在")
    if not threads[thread_id].get("pending"):
        raise HTTPException(status_code=409, detail="当前研究任务没有等待中的确认")
    payload = await request.json()
    return _stream_response(
        _research_event_stream(thread_id, resume_data=payload.get("resume", payload))
    )


@app.get("/threads/{thread_id}/state")
async def get_state(thread_id: str):
    thread = threads.get(thread_id)
    if not thread:
        return {"values": {}, "status": "missing", "pending": None, "activities": []}
    return {
        "values": thread.get("state", {}),
        "status": thread.get("status", "idle"),
        "pending": thread.get("pending"),
        "activities": thread.get("activities", []),
    }


@app.get("/providers/ollama/models")
async def ollama_models(base_url: str = "http://host.docker.internal:11434"):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
        models = [item.get("name") for item in response.json().get("models", []) if item.get("name")]
        return {"status": "connected", "base_url": base_url, "models": models}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"无法连接 Ollama：{exc}") from exc


@app.get("/ok")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=2024, reload=True)
