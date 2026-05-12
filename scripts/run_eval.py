"""Run Phase A RAGAS evaluation and produce Lab 24 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lab24_common import (
    METRICS,
    PHASE_A,
    RagOutput,
    ensure_lab24_dirs,
    heuristic_scores,
    load_corpus,
    metric_average,
    read_csv,
    run_lightweight_rag,
    write_csv,
    write_json,
)


def _parse_contexts(raw: str) -> list[str]:
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
    except Exception:
        pass
    return [raw] if raw else []


def _run_questions(testset_path: Path) -> list[RagOutput]:
    rows = read_csv(testset_path)
    corpus = load_corpus()
    outputs: list[RagOutput] = []
    for row in rows:
        answer, contexts, latency_ms = run_lightweight_rag(row["question"], corpus, top_k=5)
        if not contexts:
            contexts = _parse_contexts(row.get("contexts", ""))
        outputs.append(
            RagOutput(
                question=row["question"],
                answer=answer,
                contexts=contexts,
                ground_truth=row["ground_truth"],
                evolution_type=row.get("evolution_type", "simple"),
                latency_ms=latency_ms,
            )
        )
    return outputs


def _try_ragas(outputs: list[RagOutput]) -> list[dict] | None:
    """Run real RAGAS if packages/API are configured; return per-row metric dicts."""
    if os.getenv("LAB24_OFFLINE") == "1":
        return None
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        try:
            from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

            metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
        except Exception:
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        dataset = Dataset.from_dict({
            "question": [out.question for out in outputs],
            "answer": [out.answer for out in outputs],
            "contexts": [out.contexts for out in outputs],
            "reference": [out.ground_truth for out in outputs],
        })
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=2048)),
            embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings()),
        )
        df = result.to_pandas()
        rows: list[dict] = []
        for idx, out in enumerate(outputs):
            record = df.iloc[idx].to_dict()
            rows.append({metric: float(record.get(metric, 0) or 0) for metric in METRICS})
        return rows
    except Exception as exc:
        print(f"RAGAS unavailable, using heuristic proxy scores. Reason: {exc}")
        return None


def _build_rows(outputs: list[RagOutput], metric_rows: list[dict], scoring_mode: str) -> list[dict]:
    rows: list[dict] = []
    for out, scores in zip(outputs, metric_rows):
        row = {
            "question": out.question,
            "answer": out.answer,
            "contexts": json.dumps(out.contexts, ensure_ascii=False),
            "ground_truth": out.ground_truth,
            "evolution_type": out.evolution_type,
            "latency_ms": round(out.latency_ms, 2),
            "scoring_mode": scoring_mode,
        }
        for metric in METRICS:
            row[metric] = round(float(scores.get(metric, 0) or 0), 4)
        row["avg_score"] = round(metric_average(row), 4)
        rows.append(row)
    return rows


def _write_failure_analysis(rows: list[dict]) -> None:
    bottom = sorted(rows, key=metric_average)[:10]
    clusters = {
        "C1": {
            "name": "Missing or weak retrieval context",
            "metric": "context_recall",
            "pattern": "Questions have low context recall or context precision, usually because the first retrieved chunks do not contain enough evidence.",
            "fix": "Increase top_k, add BM25+dense hybrid search, then rerank before generation.",
        },
        "C2": {
            "name": "Answer not grounded tightly enough",
            "metric": "faithfulness",
            "pattern": "The answer overlaps weakly with retrieved context or introduces extra text beyond the evidence.",
            "fix": "Use a stricter grounded prompt, temperature 0, and refuse when evidence is missing.",
        },
        "C3": {
            "name": "Multi-context synthesis failure",
            "metric": "answer_relevancy",
            "pattern": "Multi-context questions need facts from more than one document and the answer only covers part of the request.",
            "fix": "Use query decomposition and retrieve per sub-question before final synthesis.",
        },
    }

    def assign(row: dict) -> str:
        worst = min(METRICS, key=lambda metric: float(row.get(metric, 0) or 0))
        if worst in {"context_recall", "context_precision"}:
            return "C1"
        if worst == "faithfulness":
            return "C2"
        return "C3"

    lines = [
        "# Failure Cluster Analysis",
        "",
        "## Bottom 10 Questions",
        "",
        "| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    cluster_examples: dict[str, list[str]] = {key: [] for key in clusters}
    for idx, row in enumerate(bottom, 1):
        cluster = assign(row)
        cluster_examples[cluster].append(row["question"])
        question = row["question"].replace("|", " ")[:90]
        lines.append(
            f"| {idx} | {question} | {row.get('evolution_type', '')} | "
            f"{float(row['faithfulness']):.2f} | {float(row['answer_relevancy']):.2f} | "
            f"{float(row['context_precision']):.2f} | {float(row['context_recall']):.2f} | "
            f"{float(row['avg_score']):.2f} | {cluster} |"
        )

    lines.extend(["", "## Clusters Identified", ""])
    for cid, cluster in clusters.items():
        examples = cluster_examples[cid] or [row["question"] for row in bottom[:2]]
        lines.extend([
            f"### Cluster {cid}: {cluster['name']}",
            "",
            f"**Pattern:** {cluster['pattern']}",
            "",
            "**Examples:**",
        ])
        for example in examples[:3]:
            lines.append(f"- {example}")
        lines.extend([
            "",
            f"**Root cause:** weakest metric often maps to `{cluster['metric']}`.",
            "",
            f"**Proposed fix:** {cluster['fix']}",
            "",
        ])

    (PHASE_A / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_thresholds(items: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items:
        name, _, raw = item.partition("=")
        if name and raw:
            thresholds[name] = float(raw)
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=PHASE_A / "testset_v1.csv")
    parser.add_argument("--threshold", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    ensure_lab24_dirs()
    if not args.testset.exists():
        from generate_testset import main as generate_main

        generate_main()

    outputs = _run_questions(args.testset)
    if args.max_rows:
        outputs = outputs[: args.max_rows]

    real_scores = _try_ragas(outputs)
    scoring_mode = "ragas" if real_scores is not None else "heuristic_proxy"
    metric_rows = real_scores or [heuristic_scores(out) for out in outputs]
    rows = _build_rows(outputs, metric_rows, scoring_mode)

    result_path = PHASE_A / "ragas_results.csv"
    write_csv(result_path, rows)
    summary = {
        "scoring_mode": scoring_mode,
        "num_questions": len(rows),
        "aggregate": {metric: round(sum(float(row[metric]) for row in rows) / len(rows), 4) for metric in METRICS},
        "estimated_total_cost_usd": 0.0 if scoring_mode == "heuristic_proxy" else "see OpenAI usage dashboard",
    }
    write_json(PHASE_A / "ragas_summary.json", summary)
    _write_failure_analysis(rows)

    print(f"Wrote {result_path}")
    print(f"Wrote {PHASE_A / 'ragas_summary.json'}")
    print(f"Wrote {PHASE_A / 'failure_analysis.md'}")
    print(f"Scoring mode: {scoring_mode}")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))

    thresholds = _parse_thresholds(args.threshold)
    failures = [
        f"{metric}={summary['aggregate'].get(metric, 0)} < {threshold}"
        for metric, threshold in thresholds.items()
        if float(summary["aggregate"].get(metric, 0)) < threshold
    ]
    if failures:
        print("Threshold gate failed:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
