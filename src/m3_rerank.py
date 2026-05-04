"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    """Kết quả sau khi rerank."""

    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k.

        Args:
            query: Search query.
            documents: List of {"text": str, "score": float, "metadata": dict}.
            top_k: Number of top results to return.

        Returns:
            Top-k RerankResult sorted by rerank_score descending.
        """
        if not documents:
            return []

        model = self._load_model()

        # Create query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Predict relevance scores
        scores = model.predict(pairs)

        # Handle single score (returned as scalar)
        if not hasattr(scores, "__len__"):
            scores = [scores]

        # Combine scores with documents, sort descending
        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        return [
            RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored_docs[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""

    def __init__(self) -> None:
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Rerank using flashrank (lightweight)."""
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()

            passages = [{"text": d["text"]} for d in documents]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))

            return [
                RerankResult(
                    text=r["text"],
                    original_score=documents[i].get("score", 0.0),
                    rerank_score=r.get("score", 0.0),
                    metadata=documents[i].get("metadata", {}),
                    rank=i,
                )
                for i, r in enumerate(results[:top_k])
            ]
        except ImportError:
            return []


def benchmark_reranker(
    reranker: CrossEncoderReranker | FlashrankReranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """Benchmark latency over n_runs.

    Returns:
        {"avg_ms": float, "min_ms": float, "max_ms": float}
    """
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return {
        "avg_ms": round(mean(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

    stats = benchmark_reranker(reranker, query, docs, n_runs=3)
    print(f"\nBenchmark: {stats}")
