"""Regression tests for the standalone API/frontend integration helpers."""

import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import server
from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.state import ClarifyWithUser, ResearchPlan


class ServerHelpersTest(unittest.TestCase):
    def test_langchain_messages_are_json_safe_for_frontend(self):
        payload = server._json_safe({"messages": [AIMessage(content="完成")]})

        self.assertEqual(payload["messages"][0]["role"], "assistant")
        self.assertEqual(payload["messages"][0]["type"], "ai")
        self.assertEqual(payload["messages"][0]["content"], "完成")

    def test_deepseek_is_the_runtime_default(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test", "TAVILY_API_KEY": "test"}, clear=True):
            server._validate_runtime_config({})

    def test_missing_deepseek_key_has_actionable_error(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test"}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                server._validate_runtime_config({})

    def test_native_openai_search_is_rejected_for_deepseek(self):
        config = {
            "search_api": "openai",
            "research_model": "deepseek:deepseek-chat",
            "compression_model": "deepseek:deepseek-chat",
            "final_report_model": "deepseek:deepseek-chat",
            "summarization_model": "deepseek:deepseek-chat",
        }
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True):
            with self.assertRaisesRegex(ValueError, "只能搭配 OpenAI"):
                server._validate_runtime_config(config)

    def test_multiple_search_sources_and_budget_are_parsed(self):
        config = Configuration.from_runnable_config({"configurable": {
            "search_apis": ["tavily", "duckduckgo"],
            "max_search_calls": 12,
        }})

        self.assertEqual(config.search_apis, [SearchAPI.TAVILY, SearchAPI.DUCKDUCKGO])
        self.assertEqual(config.max_search_calls, 12)

    def test_ollama_does_not_require_cloud_credentials(self):
        values = {
            "research_model": "ollama:qwen3:8b",
            "compression_model": "ollama:qwen3:8b",
            "final_report_model": "ollama:qwen3:8b",
            "summarization_model": "ollama:qwen3:8b",
            "search_apis": ["duckduckgo"],
        }
        with patch.dict(os.environ, {}, clear=True):
            parsed = server._validate_runtime_config(values)
        self.assertEqual(parsed.ollama_base_url, "http://host.docker.internal:11434")

    def test_interaction_schemas_reject_empty_content(self):
        clarification = ClarifyWithUser(
            need_clarification=True,
            intro="需要进一步确认范围",
            questions=["更关注技术还是行业应用？"],
            suggested_focus=["技术演进", "行业落地"],
            verification="",
        )
        plan = ResearchPlan(
            title="RAG 发展趋势",
            objective="分析技术演进、应用和未来瓶颈",
            sections=[
                {"title": "技术演进", "description": "检索和生成架构"},
                {"title": "行业落地", "description": "典型案例与价值"},
                {"title": "未来判断", "description": "三年趋势和限制"},
            ],
            estimated_searches=8,
        )
        self.assertTrue(clarification.need_clarification)
        self.assertEqual(len(plan.sections), 3)

    def test_state_endpoint_uses_saved_snapshot(self):
        thread_id = "test-thread"
        server.threads[thread_id] = {
            "messages": [],
            "state": {"final_report": "报告"},
            "status": "completed",
            "pending": None,
            "activities": [],
        }
        try:
            import asyncio

            result = asyncio.run(server.get_state(thread_id))
            self.assertEqual(result["values"], {"final_report": "报告"})
            self.assertEqual(result["status"], "completed")
        finally:
            server.threads.pop(thread_id, None)


if __name__ == "__main__":
    unittest.main()
