"""Standalone FastAPI server for the checkpointed deep-research graph."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import httpx
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.utils import get_api_key_for_model, get_tavily_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("open_deep_research.server")

app = FastAPI(title="Open Deep Research API")
threads: dict[str, dict[str, Any]] = {}


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
        graph_input: Any = {
            "messages": [HumanMessage(content=item.get("content", "")) for item in new_messages]
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
async def create_thread():
    thread_id = str(uuid.uuid4())
    threads[thread_id] = {
        "state": {}, "configurable": {}, "pending": None,
        "status": "idle", "activities": [],
    }
    return {"thread_id": thread_id}


@app.post("/threads/{thread_id}/runs/stream")
async def run_stream(thread_id: str, request: Request):
    return _stream_response(_research_event_stream(thread_id, input_data=await request.json()))


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
