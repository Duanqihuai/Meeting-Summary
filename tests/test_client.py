"""用本地模拟HTTP响应检查真实SDK适配，不消耗API费用、不冒充线上测试。"""
import json
import unittest
from unittest.mock import patch
import httpx
from openai import OpenAI
from common.client import Client


class ClientTests(unittest.TestCase):
    def make_client(self, handler):
        with patch.dict("os.environ", {"TEXT_API_KEY": "test-key", "TEXT_BASE_URL": "https://example.invalid/v1", "TEXT_MODEL": "test"}):
            client = Client()
        client.api.close()
        client.api = OpenAI(api_key="test-key", base_url="https://example.invalid/v1",
                            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.api.close)
        return client

    def test_text_request_and_usage(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["messages"][1]["content"], "材料")
            return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "输出"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})
        client = self.make_client(handler)
        self.assertEqual(client.text("规则", "材料"), "输出")
        self.assertEqual(client.calls[0]["usage"]["total_tokens"], 3)

    def test_stream_audio_and_empty_choices_usage(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertTrue(body["stream"])
            self.assertEqual(body["modalities"], ["text"])
            self.assertEqual(body["messages"][0]["content"][0]["type"], "input_audio")
            base = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "test"}
            rows = [dict(base, choices=[{"index": 0, "delta": {"content": "笔记"}, "finish_reason": None}]),
                    dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
                    dict(base, choices=[], usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})]
            sse = "".join("data: " + json.dumps(row) + "\n\n" for row in rows) + "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
        client = self.make_client(handler)
        self.assertEqual(client.audio("data:audio/wav;base64,AA==", "总结"), "笔记")
        self.assertEqual(client.calls[0]["usage"]["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
