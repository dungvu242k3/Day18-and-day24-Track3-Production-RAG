# AI Prompts Used

This file records the main assistant prompts and prompt templates used for Lab 24 academic integrity.

## Build Assistance

- "Đọc toàn bộ file lab24-student-edition PDF rồi lập kế hoạch làm từng phase."
- "Làm từng cái một cho đạt điểm cao nhất đề ra trong PDF."
- "Làm tiếp Phase D."

## RAG Generation Prompt

```text
Bạn là trợ lý RAG. Trả lời ngắn gọn bằng tiếng Việt, chỉ dựa trên context.
Nếu thiếu thông tin, nói rõ không tìm thấy.

Context:
{context}

Question: {question}
```

## Pairwise Judge Prompt

```text
You are an impartial evaluator. Compare two answers to the same question.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Rate based on factual accuracy, relevance, and conciseness.
Output JSON only: {"winner": "A" or "B" or "tie", "reason": "short reason"}
```

## Absolute Judge Prompt

```text
Score the answer on 4 dimensions, each 1-5 scale:
1. Factual accuracy
2. Relevance
3. Conciseness
4. Helpfulness

Question: {question}
Answer: {answer}

Output JSON only:
{"accuracy": int, "relevance": int, "conciseness": int, "helpfulness": int, "overall": float}
```

## Guardrail Refusal Copy

```text
Xin lỗi, mình không thể xử lý yêu cầu này. {reason}
```
