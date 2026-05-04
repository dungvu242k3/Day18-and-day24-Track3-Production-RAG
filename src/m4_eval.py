"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import sys
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    """Kết quả evaluation cho 1 question."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation with 4 metrics.

    Args:
        questions: List of query questions.
        answers: List of generated answers.
        contexts: List of context lists per question.
        ground_truths: List of expected answers.

    Returns:
        Dict with aggregate scores and per_question EvalResult list.
    """
    try:
        from ragas import evaluate
        from datasets import Dataset

        # ragas v0.2+: metrics are classes (capitalized)
        # ragas v0.1.x: metrics are singletons (lowercase)
        try:
            from ragas.metrics import (
                Faithfulness,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
            )

            metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
        except ImportError:
            from ragas.metrics import (  # type: ignore[no-redef]
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # ragas v0.2+ uses "reference" instead of "ground_truth"
        data_dict: dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        # Initialize LLM and Embeddings explicitly for ragas v0.2+
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # Use gpt-4o-mini with higher max_tokens to avoid "incomplete output" errors
        eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", max_tokens=2048))
        eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

        try:
            dataset = Dataset.from_dict({**data_dict, "reference": ground_truths})
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=eval_llm,
                embeddings=eval_embeddings,
            )
        except Exception:
            dataset = Dataset.from_dict({**data_dict, "ground_truth": ground_truths})
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=eval_llm,
                embeddings=eval_embeddings,
            )

        df = result.to_pandas()

        per_question: list[EvalResult] = []
        for _, row in df.iterrows():
            per_question.append(
                EvalResult(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    contexts=row.get("contexts", []),
                    ground_truth=str(row.get("reference", row.get("ground_truth", ""))),
                    faithfulness=float(row.get("faithfulness", 0.0)),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                    context_precision=float(row.get("context_precision", 0.0)),
                    context_recall=float(row.get("context_recall", 0.0)),
                )
            )

        # Calculate aggregates from DataFrame to ensure version compatibility
        def safe_mean(col: str) -> float:
            if col in df.columns:
                val = df[col].mean()
                import pandas as pd
                return float(val) if not pd.isna(val) else 0.0
            return 0.0

        return {
            "faithfulness": safe_mean("faithfulness"),
            "answer_relevancy": safe_mean("answer_relevancy"),
            "context_precision": safe_mean("context_precision"),
            "context_recall": safe_mean("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        import traceback
        print(f"  ⚠️  RAGAS evaluation error: {e}")
        traceback.print_exc()
        # Fallback: return basic scores so pipeline doesn't crash
        per_question = [
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Diagnostic mapping:
        faithfulness < 0.85     → "LLM hallucinating"      → "Tighten prompt, lower temperature"
        context_recall < 0.75   → "Missing relevant chunks" → "Improve chunking or add BM25"
        context_precision < 0.75 → "Irrelevant chunks"      → "Add reranking or metadata filter"
        answer_relevancy < 0.80  → "Answer mismatch"        → "Improve prompt template"

    Args:
        eval_results: List of EvalResult from evaluate_ragas.
        bottom_n: Number of worst questions to analyze.

    Returns:
        List of diagnosis dicts with question, worst_metric, score, diagnosis, suggested_fix.
    """
    if not eval_results:
        return []

    # Diagnostic mapping
    diagnostics = {
        "faithfulness": {
            "threshold": 0.85,
            "diagnosis": "LLM hallucinating — câu trả lời chứa thông tin không có trong context",
            "suggested_fix": "Tighten prompt: chỉ trả lời dựa trên context, lower temperature",
        },
        "context_recall": {
            "threshold": 0.75,
            "diagnosis": "Missing relevant chunks — không tìm đủ context liên quan",
            "suggested_fix": "Improve chunking hoặc thêm BM25 search, tăng top_k",
        },
        "context_precision": {
            "threshold": 0.75,
            "diagnosis": "Too many irrelevant chunks — context chứa quá nhiều noise",
            "suggested_fix": "Add reranking hoặc metadata filter để lọc context tốt hơn",
        },
        "answer_relevancy": {
            "threshold": 0.80,
            "diagnosis": "Answer doesn't match question — câu trả lời lệch chủ đề",
            "suggested_fix": "Improve prompt template, thêm instruction rõ ràng hơn",
        },
    }

    # Calculate average score for each question and sort ascending
    scored_results = []
    for result in eval_results:
        avg_score = mean([
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision,
            result.context_recall,
        ])
        scored_results.append((avg_score, result))

    scored_results.sort(key=lambda x: x[0])
    bottom_results = scored_results[:bottom_n]

    failures: list[dict] = []
    for avg_score, result in bottom_results:
        # Find worst metric
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metrics, key=lambda m: metrics[m])
        worst_score = metrics[worst_metric]

        diag = diagnostics[worst_metric]
        failures.append({
            "question": result.question,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "diagnosis": diag["diagnosis"],
            "suggested_fix": diag["suggested_fix"],
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json") -> None:
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
