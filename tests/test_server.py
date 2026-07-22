"""Regression tests for the standalone API/frontend integration helpers."""

import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import server


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

    def test_state_endpoint_uses_saved_snapshot(self):
        thread_id = "test-thread"
        server.threads[thread_id] = {
            "messages": [],
            "state": {"final_report": "报告"},
        }
        try:
            import asyncio

            result = asyncio.run(server.get_state(thread_id))
            self.assertEqual(result, {"values": {"final_report": "报告"}})
        finally:
            server.threads.pop(thread_id, None)


if __name__ == "__main__":
    unittest.main()
