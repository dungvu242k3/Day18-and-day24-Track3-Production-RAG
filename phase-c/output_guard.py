"""Phase C output guardrail wrapper for Llama Guard 3 with local fallback."""

from __future__ import annotations

import os
import re
import time


UNSAFE_PATTERNS = [
    r"how to (?:build|make|create).*(?:bomb|weapon|malware)",
    r"attack someone",
    r"self-harm",
    r"suicide",
    r"steal .*password",
    r"hack .*account",
    r"bypass .*security",
    r"derogatory",
    r"false medical advice",
]


class OutputGuard:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _fallback_check(self, agent_response: str) -> tuple[bool, str]:
        lowered = agent_response.lower()
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, lowered):
                return False, f"unsafe\nMatched fallback policy: {pattern}"
        return True, "safe\nFallback policy found no unsafe content."

    def check(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        start = time.perf_counter()
        if self.api_key:
            try:
                import requests

                payload = {
                    "model": "llama-guard-3-8b",
                    "messages": [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": agent_response},
                    ],
                    "temperature": 0,
                }
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.post(self.url, json=payload, headers=headers, timeout=20)
                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"]
                normalized = result.lower()
                is_safe = "safe" in normalized and "unsafe" not in normalized
                return is_safe, result, (time.perf_counter() - start) * 1000
            except Exception:
                pass

        is_safe, result = self._fallback_check(agent_response)
        return is_safe, result, (time.perf_counter() - start) * 1000

    async def check_async(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        import asyncio

        return await asyncio.to_thread(self.check, user_input, agent_response)
