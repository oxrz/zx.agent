"""
zAgent AI / LLM invocation module
"""

import json
from contextlib import aclosing
from typing import AsyncGenerator, List, Dict
from dataclasses import dataclass
import httpx
from utils.logger import logger


@dataclass
class LLMConfig:
    # provider/model/api_base have no defaults here; the only default source is
    # .env.example (AI_PROVIDER/AI_MODEL/AI_API_BASE), to avoid maintaining a second
    # set of hardcoded defaults duplicated in main.py.
    # Missing config should fail loudly at startup, not silently fall back to some
    # hardcoded provider.
    provider: str
    model: str
    api_base: str
    api_key: str = None
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 30.0


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._http_client = None

    @property
    def client(self):
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout), follow_redirects=True)
        return self._http_client

    async def chat(self, messages, stream=True):
        """Send a chat request. The `provider` field is only a logging label and does not
        affect dispatch -- any OpenAI-compatible /chat/completions endpoint
        (DeepSeek/OpenAI/Moonshot/etc.) works directly, no need to maintain a list of
        specific vendors in code."""
        async for chunk in self._chat_openai_like(messages, stream):
            yield chunk

    async def translate(self, text, target_lang="Chinese", stream=True):
        """Translate a single piece of text into the target language. Each call is independent, no chat history."""
        system_prompt = (
            f"You are a professional translation engine. Accurately translate the user's input, "
            f"regardless of its source language, into {target_lang}. "
            f"Output only the translation itself -- no explanations, no annotation of the original "
            f"text, no quotes, no extra content. "
            f"If the input is already in {target_lang}, return it unchanged."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        async for chunk in self.chat(messages, stream=stream):
            yield chunk

    async def _chat_openai_like(self, messages, stream):
        api_key = self.config.api_key
        if not api_key:
            yield "[Error: API key not configured]"
            return

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"

        try:
            if stream:
                async with self.client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"API request failed [{response.status_code}]: {error_text}")
                        yield f"[API error: {response.status_code}]"
                        return
                    async with aclosing(response.aiter_lines()):
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    choices = data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    continue
            else:
                response = await self.client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(f"API request failed [{response.status_code}]: {response.text}")
                    yield f"[API error: {response.status_code}]"
                    return
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                yield content
        except httpx.TimeoutException:
            logger.error("API request timed out")
            yield "[Error: request timed out]"
        except Exception as e:
            logger.error(f"API request error: {e}")
            yield f"[Error: {e}]"

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
