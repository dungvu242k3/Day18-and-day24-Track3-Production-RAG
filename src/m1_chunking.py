"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os
import sys
import glob
import re

from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_PARENT_SIZE,
    HIERARCHICAL_CHILD_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    """Đơn vị dữ liệu sau khi chunking."""

    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # 1. Split text into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences using a fast model
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    # 3. Cosine similarity helper
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # 4. Group sentences by similarity
    chunks: list[Chunk] = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            # Similarity too low → start new chunk
            chunks.append(
                Chunk(
                    text=" ".join(current_group),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                )
            )
            current_group = []
        current_group.append(sentences[i])

    # Don't forget last group
    if current_group:
        chunks.append(
            Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            )
        )

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []

    # 1. Split text into paragraphs and group into parents
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_text = ""
    p_index = 0

    for para in paragraphs:
        if len(current_text) + len(para) > parent_size and current_text:
            pid = f"parent_{p_index}"
            parents.append(
                Chunk(
                    text=current_text.strip(),
                    metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_index},
                )
            )
            # 2. Split parent into children using sliding window
            _split_parent_into_children(current_text.strip(), pid, child_size, metadata, children)
            p_index += 1
            current_text = ""
        current_text += para + "\n\n"

    # Last parent
    if current_text.strip():
        pid = f"parent_{p_index}"
        parents.append(
            Chunk(
                text=current_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_index},
            )
        )
        _split_parent_into_children(current_text.strip(), pid, child_size, metadata, children)

    return parents, children


def _split_parent_into_children(
    parent_text: str,
    parent_id: str,
    child_size: int,
    metadata: dict,
    children: list[Chunk],
) -> None:
    """Split a parent chunk into smaller children using character-based sliding window."""
    start = 0
    c_index = 0
    while start < len(parent_text):
        end = min(start + child_size, len(parent_text))
        child_text = parent_text[start:end].strip()
        if child_text:
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                        "child_index": c_index,
                    },
                    parent_id=parent_id,
                )
            )
            c_index += 1
        start = end


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    chunks: list[Chunk] = []

    # 1. Split by markdown headers (h1-h3)
    sections = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    # 2. Pair headers with their content
    current_header = ""
    current_content = ""
    for part in sections:
        if re.match(r"^#{1,3}\s+", part):
            # Save previous section
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "section": current_header.strip(),
                            "strategy": "structure",
                            "chunk_index": len(chunks),
                        },
                    )
                )
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Don't forget last section
    if current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "section": current_header.strip(),
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            )
        )

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results: dict[str, dict] = {}

    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_parents: list[Chunk] = []
    all_children: list[Chunk] = []
    all_structure: list[Chunk] = []

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})

        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))

        parents, children = chunk_hierarchical(text, metadata=meta)
        all_parents.extend(parents)
        all_children.extend(children)

        all_structure.extend(chunk_structure_aware(text, metadata=meta))

    def _stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    results["basic"] = _stats(all_basic)
    results["semantic"] = _stats(all_semantic)
    results["hierarchical"] = {
        "parents": _stats(all_parents),
        "children": _stats(all_children),
    }
    results["structure"] = _stats(all_structure)

    # Print comparison table
    print(f"\n{'Strategy':<18} {'Chunks':>8} {'Avg Len':>8} {'Min':>6} {'Max':>6}")
    print("-" * 50)
    for name in ["basic", "semantic", "structure"]:
        s = results[name]
        print(f"{name:<18} {s['num_chunks']:>8} {s['avg_length']:>8} {s['min_length']:>6} {s['max_length']:>6}")
    h = results["hierarchical"]
    p_s, c_s = h["parents"], h["children"]
    print(f"{'hierarchical(P)':<18} {p_s['num_chunks']:>8} {p_s['avg_length']:>8} {p_s['min_length']:>6} {p_s['max_length']:>6}")
    print(f"{'hierarchical(C)':<18} {c_s['num_chunks']:>8} {c_s['avg_length']:>8} {c_s['min_length']:>6} {c_s['max_length']:>6}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
