"""Shared utilities for Lab 24 evaluation, judging, and guardrail scripts."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PHASE_A = ROOT / "phase-a"
PHASE_B = ROOT / "phase-b"
PHASE_C = ROOT / "phase-c"
PHASE_D = ROOT / "phase-d"
REPORTS = ROOT / "reports"
TEST_SET_JSON = ROOT / "test_set.json"


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class RagOutput:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    evolution_type: str = "simple"
    latency_ms: float = 0.0


def ensure_lab24_dirs() -> None:
    for path in [PHASE_A, PHASE_B, PHASE_C, PHASE_D, REPORTS, ROOT / "scripts"]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = fieldnames or []
    else:
        fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_base_qa() -> list[dict[str, str]]:
    return read_json(TEST_SET_JSON)  # type: ignore[return-value]


def load_corpus() -> list[dict[str, str]]:
    """Load markdown and PDF corpus into document records."""
    docs: list[dict[str, str]] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})

    for path in sorted(DATA_DIR.glob("*.pdf")):
        try:
            import fitz

            pdf = fitz.open(path)
            for page_no, page in enumerate(pdf, 1):
                text = page.get_text("text").strip()
                if text:
                    docs.append({"source": f"{path.name}:p{page_no}", "text": text})
        except Exception:
            continue
    return docs


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


def overlap_score(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / math.sqrt(len(a_tokens) * len(b_tokens))


def keyword_retrieve(question: str, corpus: list[dict[str, str]], top_k: int = 5) -> list[str]:
    scored = []
    q_terms = Counter(tokenize(question))
    for doc in corpus:
        terms = Counter(tokenize(doc["text"]))
        score = sum(min(freq, terms.get(tok, 0)) for tok, freq in q_terms.items())
        score += overlap_score(question, doc["text"])
        scored.append((score, doc["text"]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for score, text in scored[:top_k] if score > 0]


def openai_answer(question: str, contexts: list[str]) -> str | None:
    if os.getenv("LAB24_OFFLINE") == "1" or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI(timeout=20)
        context = "\n\n".join(contexts[:5])
        response = client.chat.completions.create(
            model=os.getenv("LAB24_GENERATION_MODEL", "gpt-4o-mini"),
            temperature=0.0,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý RAG. Trả lời ngắn gọn bằng tiếng Việt, "
                        "chỉ dựa trên context. Nếu thiếu thông tin, nói rõ không tìm thấy."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def extractive_answer(question: str, contexts: list[str]) -> str:
    """Cheap fallback answer: pick the most query-overlapping sentences from context."""
    sentences: list[str] = []
    for ctx in contexts[:3]:
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?。])\s+|\n", ctx) if s.strip())
    if not sentences:
        return "Không tìm thấy thông tin trong tài liệu."
    ranked = sorted(sentences, key=lambda s: overlap_score(question, s), reverse=True)
    return " ".join(ranked[:2])[:900]


def run_lightweight_rag(question: str, corpus: list[dict[str, str]], top_k: int = 5) -> tuple[str, list[str], float]:
    start = time.perf_counter()
    contexts = keyword_retrieve(question, corpus, top_k=top_k)
    answer = openai_answer(question, contexts) or extractive_answer(question, contexts)
    latency_ms = (time.perf_counter() - start) * 1000
    return answer, contexts, latency_ms


def heuristic_scores(output: RagOutput) -> dict[str, float]:
    """Deterministic proxy scores used only when RAGAS cannot run locally."""
    joined_context = "\n".join(output.contexts)
    answer_gt = overlap_score(output.answer, output.ground_truth)
    answer_question = overlap_score(output.answer, output.question)
    context_gt = max((overlap_score(ctx, output.ground_truth) for ctx in output.contexts), default=0.0)
    context_question = max((overlap_score(ctx, output.question) for ctx in output.contexts), default=0.0)

    faithfulness = min(1.0, 0.35 + 0.65 * overlap_score(output.answer, joined_context))
    answer_relevancy = min(1.0, 0.30 + 0.45 * answer_gt + 0.25 * answer_question)
    context_precision = min(1.0, 0.25 + 0.75 * context_question)
    context_recall = min(1.0, 0.25 + 0.75 * context_gt)

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
    }


def metric_average(row: dict) -> float:
    vals = [float(row.get(metric, 0) or 0) for metric in METRICS]
    return sum(vals) / len(vals)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[int(idx)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def truncate(text: str, max_len: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
