"""OpenAI 兼容协议客户端：仅配置地址即可切换 DeepSeek/Qwen。"""
import os
import time
from urllib.parse import urlparse


class Client:
    def __init__(self, prefix="TEXT"):
        from openai import OpenAI
        key = os.getenv(prefix + "_API_KEY")
        base = os.getenv(prefix + "_BASE_URL")
        self.model = os.getenv(prefix + "_MODEL")
        if not all((key, base, self.model)):
            raise ValueError(f"请在 .env 配置 {prefix}_API_KEY、{prefix}_BASE_URL、{prefix}_MODEL")
        self.base = base
        self.api = OpenAI(api_key=key, base_url=base, timeout=180, max_retries=2)
        self.calls = []

    def text(self, system, text):
        start = time.perf_counter()
        options = {}
        # DeepSeek当前模型可能默认启用推理；结构化摘要关闭推理，避免预算耗尽却无正文。
        if (urlparse(self.base).hostname or "").endswith("deepseek.com"):
            options["extra_body"] = {"thinking": {"type": "disabled"}}
        response = self.api.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": text}],
            max_tokens=4096, **options)
        content = response.choices[0].message.content or ""
        self.calls.append({"model": self.model, "response_model": response.model, "request_id": response.id,
                           "seconds": time.perf_counter() - start, "finish_reason": response.choices[0].finish_reason,
                           "thinking": "disabled" if options else "provider_default",
                           "response": content, "usage": response.usage.model_dump() if response.usage else None})
        if response.choices[0].finish_reason != "stop" or not content.strip():
            raise ValueError("文本模型未正常完成或返回空内容，请检查原始响应与输出预算")
        return content

    def audio(self, data_uri, prompt):
        start = time.perf_counter()
        stream = self.api.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": data_uri, "format": "wav"}},
                {"type": "text", "text": prompt}]}],
            modalities=["text"], stream=True, stream_options={"include_usage": True}, max_tokens=4096)
        parts, usage, finish = [], None, None
        with stream:
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                if chunk.choices:
                    parts.append(chunk.choices[0].delta.content or "")
                    finish = chunk.choices[0].finish_reason or finish
        content = "".join(parts)
        self.calls.append({"model": self.model, "seconds": time.perf_counter() - start,
                           "response": content, "usage": usage})
        if finish != "stop" or not content.strip():
            raise ValueError(f"语音模型流未正常完成：{finish}")
        return content
