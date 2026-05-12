"""Compute Cohen's kappa for Phase B human calibration."""

from __future__ import annotations

from pathlib import Path

from lab24_common import PHASE_B, read_csv


def _cohen_kappa(human: list[str], judge: list[str]) -> float:
    labels = sorted(set(human) | set(judge))
    n = len(human)
    if n == 0:
        return 0.0
    observed = sum(1 for h, j in zip(human, judge) if h == j) / n
    human_counts = {label: human.count(label) for label in labels}
    judge_counts = {label: judge.count(label) for label in labels}
    expected = sum((human_counts[label] / n) * (judge_counts[label] / n) for label in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def _interpret(kappa: float) -> str:
    if kappa < 0:
        return "Worse than chance - check label mapping and judge prompt."
    if kappa < 0.2:
        return "Slight agreement - not reliable enough for monitoring."
    if kappa < 0.4:
        return "Fair agreement - judge has likely bias or rubric mismatch."
    if kappa < 0.6:
        return "Moderate agreement - usable for monitoring with more samples."
    if kappa < 0.8:
        return "Substantial agreement - production-ready calibration signal."
    return "Almost perfect agreement."


def main() -> None:
    human_rows = read_csv(PHASE_B / "human_labels.csv")
    pairwise_rows = read_csv(PHASE_B / "pairwise_results.csv")
    judge_by_id = {row["question_id"]: row["winner_after_swap"] for row in pairwise_rows}

    human = [row["human_winner"] for row in human_rows]
    judge = [judge_by_id.get(row["question_id"], "tie") for row in human_rows]
    kappa = _cohen_kappa(human, judge)
    interpretation = _interpret(kappa)

    lines = [
        "# Cohen's Kappa Analysis",
        "",
        f"- Samples: {len(human)}",
        f"- Cohen's kappa: {kappa:.3f}",
        f"- Interpretation: {interpretation}",
        "",
    ]
    if kappa < 0.6:
        lines.extend([
            "## Root Cause Analysis",
            "",
            "Agreement is below the production-ready target. Likely causes are a small calibration sample, answer length preference, or the heuristic fallback judge being less nuanced than a human evaluator.",
            "",
            "Next action: label 20 additional pairs, keep swap-and-average, and tighten the judge rubric with explicit factuality and concision anchors.",
        ])
    else:
        lines.extend([
            "## Root Cause Analysis",
            "",
            "No remediation required for the current sample. Continue periodic calibration as the corpus and prompt versions change.",
        ])

    output = PHASE_B / "kappa_analysis.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
