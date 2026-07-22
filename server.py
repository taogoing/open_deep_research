"""FastAPI server wrapping the Open Deep Research LangGraph agent.

Replaces ``langgraph dev`` with a standalone FastAPI + uvicorn server
that provides the same SSE streaming API the frontend expects.
"""

import json
import logging
import uuid
from typing import Any, AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.utils import get_api_key_for_model, get_tavily_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("open_deep_research.server")

app = FastAPI(title="Open Deep Research API")

# ── In-memory thread storage ──────────────────────────────────
threads: dict[str, dict[str, Any]] = {}


# ── Helpers ────────────────────────────────────────────────────

def _to_langchain_message(m: dict[str, Any]):
    """Convert a frontend role-based message dict into a LangChain message."""
    role = m.get("role", "")
    content = m.get("content", "")
    if role in ("human", "user"):
        return HumanMessage(content=content)
    if role in ("ai", "assistant"):
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _message_to_wire(message: BaseMessage | dict[str, Any]) -> dict[str, Any]:
    """Convert LangChain messages to the compact shape used by the frontend."""
    if isinstance(message, dict):
        return message
    role = "assistant" if message.type in ("ai", "assistant") else "user"
    return {"role": role, "type": message.type, "content": message.content}


def _json_safe(value: Any) -> Any:
    """Recursively make LangGraph state safe for JSON/SSE transport."""
    if isinstance(value, BaseMessage):
        return _message_to_wire(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _validate_runtime_config(configurable: dict[str, Any]) -> None:
    """Fail early with actionable credential/provider errors."""
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

    search_api = config.search_api
    if search_api == SearchAPI.TAVILY and not get_tavily_api_key(runnable_config):
        raise ValueError("Tavily 搜索缺少凭据，请在 .env 中设置 TAVILY_API_KEY 后重启服务。")
    if search_api == SearchAPI.OPENAI and not config.research_model.startswith("openai:"):
        raise ValueError("OpenAI 原生搜索只能搭配 OpenAI 研究模型；DeepSeek 请使用 Tavily 或 MCP 搜索。")
    if search_api == SearchAPI.ANTHROPIC and not config.research_model.startswith("anthropic:"):
        raise ValueError("Anthropic 原生搜索只能搭配 Anthropic 研究模型；DeepSeek 请使用 Tavily 或 MCP 搜索。")


def _friendly_error(exc: Exception) -> str:
    """Avoid exposing a provider traceback when a concise fix is available."""
    message = str(exc)
    lowered = message.lower()
    if "missing credentials" in lowered and "openai" in lowered:
        return "OpenAI 模型缺少凭据。请改用 deepseek:deepseek-chat，或设置 OPENAI_API_KEY。"
    if "authentication" in lowered or "api key" in lowered:
        return f"模型或搜索服务认证失败：{message}"
    return message


def _extract_streaming_text(chunk: Any) -> str | None:
    """Extract the text delta from a streaming chunk across all major providers."""
    if hasattr(chunk, "content") and isinstance(chunk.content, str):
        return chunk.content or None

    if hasattr(chunk, "content") and isinstance(chunk.content, list):
        for block in chunk.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    return text

    if hasattr(chunk, "additional_kwargs"):
        ak = chunk.additional_kwargs or {}
        if isinstance(ak, dict):
            block = ak.get("content_block") or {}
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    return text

    return None


# Node names the frontend maps to step-indicator labels
_STEP_NODES = {
    "clarify_with_user",
    "write_research_brief",
    "research_supervisor",
    "final_report_generation",
}


def _match_step_node(name: str) -> str | None:
    """Return the canonical step-node name if *name* matches one."""
    if not name:
        return None
    # exact match
    if name in _STEP_NODES:
        return name
    # subgraph internal nodes often appear as "research_supervisor:supervisor" etc.
    for step in _STEP_NODES:
        if name.startswith(step):
            return step
    return None


# ── SSE event generator ────────────────────────────────────────

async def _research_event_stream(
    thread_id: str, input_data: dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Execute the Deep Researcher graph and yield SSE events."""
    new_messages = input_data.get("input", {}).get("messages", [])
    thread = threads.setdefault(thread_id, {"messages": [], "state": {}})
    thread["messages"].extend(new_messages)

    configurable = input_data.get("config", {}).get("configurable", {})
    run_config: dict[str, Any] = {"configurable": configurable}

    input_state = {
        "messages": [_to_langchain_message(m) for m in thread["messages"]]
    }
    active_step: str | None = None
    latest_state: dict[str, Any] = dict(thread.get("state", {}))

    try:
        _validate_runtime_config(configurable)
        async for event in deep_researcher.astream_events(
            input_state, run_config, version="v2"
        ):
            kind: str = event.get("event", "")
            name: str = event.get("name", "")
            data: dict[str, Any] = event.get("data", {})

            # ── Node progress → updates ─────────────────────────
            if kind == "on_chain_start":
                matched = _match_step_node(name)
                if matched:
                    active_step = matched
                    logger.info("Step start: %s (raw name=%s)", matched, name)
                    yield (
                        f"event: updates\n"
                        f"data: {json.dumps({matched: {'status': 'active'}})}\n\n"
                    )

            # ── Token streaming → messages/partial ──────────────
            metadata = event.get("metadata", {}) or {}
            stream_node = _match_step_node(str(metadata.get("langgraph_node", "")))
            if kind == "on_chat_model_stream" and stream_node == "final_report_generation":
                chunk = data.get("chunk")
                if chunk is not None:
                    text = _extract_streaming_text(chunk)
                    if text:
                        payload = [[
                            "messages/partial",
                            [{"content": text, "type": "AIMessageChunk"}],
                        ]]
                        yield (
                            f"event: messages/partial\n"
                            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        )

            # ── Node output → values ────────────────────────────
            if kind == "on_chain_end":
                matched = _match_step_node(name)
                if matched:
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        safe_output = _json_safe(output)
                        latest_state.update(safe_output)
                        logger.info(
                            "Step end: %s  keys=%s", matched, list(output.keys())[:8]
                        )
                        yield (
                            f"event: values\n"
                            f"data: {json.dumps(safe_output, ensure_ascii=False)}\n\n"
                        )
                    yield (
                        f"event: updates\n"
                        f"data: {json.dumps({matched: {'status': 'done'}})}\n\n"
                    )
                    if active_step == matched:
                        active_step = None

                # The root graph end contains the most complete state snapshot.
                root_output = data.get("output")
                if isinstance(root_output, dict) and (
                    "messages" in root_output or "final_report" in root_output
                ):
                    latest_state.update(_json_safe(root_output))

        if latest_state.get("messages"):
            thread["messages"] = latest_state["messages"]
        thread["state"] = latest_state
        yield f"event: end\ndata: {json.dumps(None)}\n\n"

    except Exception as exc:
        logger.exception("Research stream error")
        failed_step = active_step or "clarify_with_user"
        yield (
            f"event: updates\n"
            f"data: {json.dumps({failed_step: {'status': 'failed'}})}\n\n"
        )
        yield f"event: error\ndata: {json.dumps(_friendly_error(exc), ensure_ascii=False)}\n\n"


# ── Routes ─────────────────────────────────────────────────────

@app.post("/threads")
async def create_thread():
    """Create a new conversation thread."""
    thread_id = str(uuid.uuid4())
    threads[thread_id] = {"messages": [], "state": {}}
    return {"thread_id": thread_id}


@app.post("/threads/{thread_id}/runs/stream")
async def run_stream(thread_id: str, request: Request):
    """Stream a research run as SSE."""
    input_data: dict[str, Any] = await request.json()
    return StreamingResponse(
        _research_event_stream(thread_id, input_data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/threads/{thread_id}/state")
async def get_state(thread_id: str):
    """Return the latest state snapshot for a thread."""
    thread = threads.get(thread_id)
    return {"values": thread.get("state", {}) if thread else {}}


@app.get("/ok")
async def health():
    """Health-check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=2024, reload=True)
