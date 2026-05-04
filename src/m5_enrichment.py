"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

# Rate limit: tối thiểu khoảng cách giữa các API call (giây)
_MIN_API_INTERVAL = 0.5
_last_api_call = 0.0


def _rate_limit() -> None:
    """Simple rate limiter để tránh OpenAI rate limit."""
    global _last_api_call
    now = time.time()
    elapsed = now - _last_api_call
    if elapsed < _MIN_API_INTERVAL:
        time.sleep(_MIN_API_INTERVAL - elapsed)
    _last_api_call = time.time()


def _get_openai_client():
    """Lazy OpenAI client initialization with API key validation."""
    if not OPENAI_API_KEY:
        return None
    from openai import OpenAI

    return OpenAI()


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """Tạo summary ngắn cho chunk.

    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    client = _get_openai_client()
    if client:
        try:
            _rate_limit()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  Summarize API error: {e}")

    # Extractive fallback (không cần API)
    sentences = text.split(". ")
    return ". ".join(sentences[:2]).strip() + ("." if not sentences[0].endswith(".") else "")


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate câu hỏi mà chunk có thể trả lời.

    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    client = _get_openai_client()
    if client:
        try:
            _rate_limit()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên 1 dòng, không đánh số."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.5,
            )
            raw = resp.choices[0].message.content.strip()
            questions = [
                q.strip().lstrip("0123456789.-) ")
                for q in raw.split("\n")
                if q.strip()
            ]
            return questions[:n_questions]
        except Exception as e:
            print(f"  ⚠️  HyQA API error: {e}")

    # Fallback: generate basic questions from text
    return [f"Nội dung chính của đoạn văn này là gì?"]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend context giải thích chunk nằm ở đâu trong document.

    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    client = _get_openai_client()
    if client:
        try:
            _rate_limit()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu "
                            "và nói về chủ đề gì. Chỉ trả về 1 câu duy nhất."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=80,
                temperature=0.3,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  Contextual prepend API error: {e}")

    # Fallback: simple title-based prepend
    if document_title:
        return f"Trích từ tài liệu: {document_title}\n\n{text}"
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """LLM extract metadata tự động: topic, entities, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    client = _get_openai_client()
    if client:
        try:
            _rate_limit()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Trích xuất metadata từ đoạn văn. "
                            'Trả về JSON thuần: {"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance|legal", "language": "vi|en"}'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            # Clean markdown code block markers if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  ⚠️  Metadata extraction error: {e}")

    # Fallback: basic metadata
    return {
        "topic": "unknown",
        "entities": [],
        "category": "unknown",
        "language": "vi",
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched: list[EnrichedChunk] = []
    use_all = "full" in methods

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        summary = ""
        questions: list[str] = []
        enriched_text = text
        auto_meta: dict = {}

        if use_all or "summary" in methods:
            summary = summarize_chunk(text)

        if use_all or "hyqa" in methods:
            questions = generate_hypothesis_questions(text)

        if use_all or "contextual" in methods:
            enriched_text = contextual_prepend(text, document_title=source)

        if use_all or "metadata" in methods:
            auto_meta = extract_metadata(text)

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**chunk.get("metadata", {}), **auto_meta},
                method="+".join(methods),
            )
        )

        if (i + 1) % 5 == 0:
            print(f"    Enriched {i + 1}/{len(chunks)} chunks...")

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
