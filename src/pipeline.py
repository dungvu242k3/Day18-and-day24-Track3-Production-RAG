"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4+M5."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import FlashrankReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY

# Latency tracking
_latency: dict[str, float] = {}


def _track(label: str, start: float) -> None:
    """Record elapsed time for a pipeline step."""
    _latency[label] = round(time.time() - start, 2)


def build_pipeline() -> tuple:
    """Build production RAG pipeline: M1 → M5 → M2 → M3."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/4] Chunking documents...")
    t = time.time()
    docs = load_documents()
    all_chunks: list[dict] = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })
    _track("chunking", t)
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents [{_latency['chunking']}s]")

    # Step 2: Enrichment (M5) — BONUS
    print("\n[2/4] Enriching chunks (M5)...")
    t = time.time()
    enriched = enrich_chunks(all_chunks, methods=["contextual"])
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  Enriched {len(enriched)} chunks [{time.time() - t:.1f}s]")
    else:
        print("  ⚠️  M5 not implemented — using raw chunks (fallback)")
    _track("enrichment", t)

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    t = time.time()
    search = HybridSearch()
    search.index(all_chunks)
    _track("indexing", t)
    print(f"  Indexed [{_latency['indexing']}s]")

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    t = time.time()
    reranker = FlashrankReranker()
    _track("reranker_load", t)

    return search, reranker


def run_query(
    query: str,
    search: HybridSearch,
    reranker: FlashrankReranker,
) -> tuple[str, list[str]]:
    """Run single query through pipeline: Search → Rerank → Generate."""
    # Search
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    # Rerank
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    # LLM Generation — cải thiện Faithfulness score
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Trả lời câu hỏi CHỈ dựa trên context được cung cấp. "
                            "Nếu context không chứa đủ thông tin, trả lời 'Không tìm thấy thông tin trong tài liệu.' "
                            "Trả lời ngắn gọn, chính xác bằng tiếng Việt."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}",
                    },
                ],
                max_tokens=300,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip(), contexts
        except Exception as e:
            print(f"  ⚠️  LLM generation error: {e}")

    # Fallback: return first context as answer
    answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: FlashrankReranker) -> dict:
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    t_eval = time.time()
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        t_q = time.time()
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i + 1}/{len(test_set)}] {item['question'][:50]}... [{time.time() - t_q:.1f}s]")

    _track("query_total", t_eval)

    print("\n[Eval] Running RAGAS...")
    t = time.time()
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    _track("ragas_eval", t)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    # Latency breakdown — BONUS (+2đ)
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    for step, elapsed in _latency.items():
        print(f"  {step:<20} {elapsed:>8.2f}s")
    print(f"  {'TOTAL':<20} {sum(_latency.values()):>8.2f}s")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
