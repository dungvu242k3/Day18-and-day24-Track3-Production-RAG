"""Phase C input guardrails: PII redaction, topic validation, and injection checks."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass


VN_PII = {
    "cccd": r"\b\d{12}\b",
    "phone_vn": r"\b(?:\+84|0)\d{9,10}\b",
    "tax_code": r"\b\d{10}(?:-\d{3})?\b",
    "email": r"\b[\w.\-+]+@[\w.\-]+\.\w+\b",
}

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"pretend you are",
    r"\bDAN\b",
    r"jailbreak",
    r"without restrictions",
    r"no guidelines",
    r"decode this base64",
    r"system prompt",
    r"developer message",
    r"roleplay.*evil",
    r"how to hack",
]


@dataclass
class GuardResult:
    ok: bool
    text: str
    reason: str
    latency_ms: float


class InputGuard:
    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        except Exception:
            self._analyzer = None
            self._anonymizer = None

    def scrub_vn(self, text: str) -> tuple[str, bool]:
        found = False
        for name, pattern in VN_PII.items():
            text, count = re.subn(pattern, f"[{name.upper()}]", text)
            found = found or count > 0
        return text, found

    def scrub_ner(self, text: str) -> tuple[str, bool]:
        if not self._analyzer or not self._anonymizer or not text:
            return text, False
        try:
            results = self._analyzer.analyze(text=text, language="en")
            anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results).text
            return anonymized, anonymized != text
        except Exception:
            return text, False

    def detect_injection(self, text: str) -> tuple[bool, str]:
        lowered = text.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                return True, f"Prompt injection pattern matched: {pattern}"
        return False, ""

    def sanitize(self, text: str) -> tuple[str, float, bool]:
        start = time.perf_counter()
        text = text or ""
        scrubbed, vn_found = self.scrub_vn(text)
        scrubbed, ner_found = self.scrub_ner(scrubbed)
        latency_ms = (time.perf_counter() - start) * 1000
        return scrubbed, latency_ms, vn_found or ner_found

    async def sanitize_async(self, text: str) -> tuple[str, float, bool]:
        return await asyncio.to_thread(self.sanitize, text)

    def check(self, text: str) -> GuardResult:
        start = time.perf_counter()
        sanitized, _, pii_found = self.sanitize(text)
        injection_found, reason = self.detect_injection(sanitized)
        latency_ms = (time.perf_counter() - start) * 1000
        if injection_found:
            return GuardResult(False, sanitized, reason, latency_ms)
        return GuardResult(True, sanitized, "PII redacted" if pii_found else "OK", latency_ms)


class TopicGuard:
    def __init__(self, allowed_topics: list[str] | None = None) -> None:
        self.allowed_topics = allowed_topics or [
            "RAG evaluation",
            "guardrails",
            "personal data protection",
            "Vietnamese privacy regulation",
            "company HR policy",
            "company IT security",
            "financial reporting",
            "DHA Surfaces",
        ]
        self.allowed_keywords = {
            "rag",
            "ragas",
            "eval",
            "guard",
            "guardrail",
            "llm",
            "dữ liệu",
            "du lieu",
            "nghị định",
            "nghi dinh",
            "cccd",
            "bảo vệ",
            "bao ve",
            "nhân viên",
            "nhan vien",
            "nghỉ phép",
            "nghi phep",
            "mật khẩu",
            "mat khau",
            "vpn",
            "thuế",
            "thue",
            "doanh thu",
            "dha",
            "surfaces",
        }

    def check(self, text: str) -> tuple[bool, str]:
        lowered = (text or "").lower()
        hits = [keyword for keyword in self.allowed_keywords if keyword in lowered]
        if hits:
            return True, f"On topic: matched {', '.join(sorted(hits)[:3])}"
        return False, "Mình chỉ hỗ trợ câu hỏi trong phạm vi RAG evaluation, guardrails, dữ liệu cá nhân, HR/IT policy và báo cáo tài chính của corpus lab."

    async def check_async(self, text: str) -> tuple[bool, str]:
        return await asyncio.to_thread(self.check, text)
