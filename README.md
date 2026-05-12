# Lab 24 — Full Evaluation & Guardrail System

## Overview

This repository extends the Day 18 Vietnamese Production RAG pipeline into a Lab 24 evaluation and guardrail stack. It includes synthetic test-set generation, RAGAS evaluation, LLM-as-Judge calibration, input/output guardrails, adversarial testing, latency benchmarks, and a production blueprint.

The system uses the existing Day 18 corpus and RAG implementation as the base application, then adds production-readiness layers around it: CI eval gates, failure clustering, judge bias checks, PII redaction, topic validation, prompt-injection checks, output safety checks, and async audit logging.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env` for real RAGAS and LLM-as-Judge runs. Optional: set `GROQ_API_KEY` for Llama Guard 3 API output checks. Without these keys, scripts keep fallback behavior for local smoke testing, but real scoring is preferred for submission.

## Lab 24 Results Summary

### Phase A: RAGAS

- Test set: 50 questions, distributed as 25 simple, 12 reasoning, 13 multi-context.
- Faithfulness: 0.9556
- Answer Relevancy: 0.8437
- Context Precision: 0.9100
- Context Recall: 1.0000
- Results: `phase-a/ragas_results.csv`
- Failure analysis: `phase-a/failure_analysis.md`
- Cost: see OpenAI usage dashboard for exact run cost.

### Phase B: LLM-as-Judge

- Pairwise judge: 30 questions with swap-and-average.
- Absolute scoring: 4 dimensions, 1-5 scale.
- Cohen's kappa vs calibration labels: 1.000 on 10 samples.
- Bias report: `phase-b/judge_bias_report.md`

### Phase C: Guardrails

- Topic validator accuracy: 95%
- Topic refuse rate: 55% on the balanced topic test set.
- Adversarial detection rate: 100% on 20 attacks.
- False positive rate: 10% on 10 legitimate queries.
- L1 Input Guard P95: 8.221 ms
- L3 Output Guard P95: 0.651 ms
- Total P95: 9184.899 ms, bottleneck is L2 RAG/LLM generation.

### Phase D: Blueprint

- Production blueprint: `phase-d/blueprint.md`
- Includes SLOs, architecture diagram, alert playbooks, and monthly cost estimate.

## Commands

```bash
python scripts/generate_testset.py --size 50
python scripts/run_eval.py
python scripts/run_judge.py --limit 30
python scripts/kappa_analysis.py
python phase-c/full_pipeline.py
```

For offline smoke tests:

```bash
set LAB24_OFFLINE=1
python scripts/run_eval.py
```

## Demo Video Checklist

1. Show `phase-a/ragas_summary.json` and a short eval run.
2. Show `phase-b/pairwise_results.csv` and `phase-b/kappa_analysis.md`.
3. Show guardrail artifacts: PII, adversarial tests, latency benchmark.
4. Show `phase-d/blueprint.md` architecture and SLO table.

---

# Lab 18: Production RAG Pipeline

**AICB-P2T3 · Ngày 18 · Production RAG**  
**Giảng viên:** M.Sc Trần Minh Tú · **Thời gian:** 2 giờ

---

## Tổng quan

Lab gồm **2 phần**:

| Phần | Hình thức | Thời gian | Mô tả |
|------|-----------|-----------|-------|
| **Phần A** | Cá nhân | 1.5 giờ | Implement 1 trong 4 modules |
| **Phần B** | Nhóm (3–4 người) | 30 phút | Ghép modules → full pipeline → eval → present |

```
  Cá nhân                         Nhóm
  ┌────────────┐
  │ M1 Chunking│──┐
  ├────────────┤  │    ┌──────────────────────────────┐
  │ M2 Search  │──┼───▶│  Production RAG System        │
  ├────────────┤  │    │  pipeline.py + RAGAS eval     │
  │ M3 Rerank  │──┤    │  + failure analysis           │
  ├────────────┤  │    └──────────────────────────────┘
  │ M4 Eval    │──┘
  └────────────┘
```

## Quick Start

```bash
git clone <repo-url> && cd lab18-production-rag
docker compose up -d                    # Qdrant
pip install -r requirements.txt
cp .env.example .env                    # Điền API keys
python naive_baseline.py                # ⚠️ Chạy TRƯỚC để có baseline
```

## Chạy toàn bộ

```bash
python main.py                          # Naive + Production + So sánh
python check_lab.py                     # Kiểm tra trước khi nộp
```

## Cấu trúc repo

```
lab18-production-rag/
├── README.md                   # File này
├── ASSIGNMENT_INDIVIDUAL.md    # ★ Đề bài cá nhân (Phần A)
├── ASSIGNMENT_GROUP.md         # ★ Đề bài nhóm (Phần B)
├── RUBRIC.md                   # Hệ thống chấm điểm chi tiết
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies (pinned)
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Sample corpus tiếng Việt
│   ├── sample_01.md
│   ├── sample_02.md
│   └── sample_03.md
├── test_set.json               # 20 Q&A pairs
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   └── pipeline.py             # Ghép nhóm
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   └── test_m4.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (nhóm)
│   ├── group_report.md         # Báo cáo nhóm
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (sau khi chạy main.py)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    ├── failure_analysis.md
    └── group_report.md
```

## Timeline

| Thời gian | Hoạt động |
|-----------|-----------|
| 0:00–0:15 | Setup + chạy `naive_baseline.py` |
| 0:15–1:45 | **Phần A (cá nhân):** implement module → `pytest tests/test_m*.py` |
| 1:45–2:15 | **Phần B (nhóm):** ghép → `python src/pipeline.py` → failure analysis |
| 2:15–2:30 | Presentation 5 phút/nhóm |
