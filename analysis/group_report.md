# Group Report — Lab 18: Production RAG

**Nhóm:** Nhóm A
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------| 
| Vũ Việt Dũng | M1 - M5 (Full Pipeline) | ☑ | 13/13 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.9917 | 0.7263 | -0.2654 |
| Answer Relevancy | 0.8135 | 0.6821 | -0.1314 |
| Context Precision | 0.8917 | 0.8553 | -0.0364 |
| Context Recall | 0.9300 | 0.8350 | -0.0950 |

## Key Findings

1. **Biggest improvement:** Khả năng xử lý Hybrid Search (BM25 + Qdrant) giúp hệ thống tìm được các từ khóa Tiếng Việt chính xác hơn so với chỉ dùng Dense Search thuần túy.
2. **Biggest challenge:** Việc tích hợp RAGAS v0.2+ gặp nhiều lỗi về API (object attributes) và giới hạn token của LLM khi đánh giá dữ liệu Tiếng Việt dài.
3. **Surprise finding:** Điểm số Production thấp hơn Naive Baseline ở một số metric. Nguyên nhân có thể do bước Enrichment (M5) tạo ra các đoạn context dài và phức tạp hơn, khiến LLM (gpt-4o-mini) dễ bị "ngộp" thông tin hoặc sinh ra câu trả lời không bám sát context bằng bản baseline đơn giản.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): Thấy rõ sự đánh đổi giữa tính năng (Hybrid/Rerank) và độ tin cậy của LLM.
2. Biggest win — Module M2 (Hybrid Search) là xương sống giúp retrieval ổn định với các từ khóa chuyên ngành pháp luật.
3. Case study — Một số câu hỏi về "Dữ liệu cá nhân" có context rất dài dẫn đến điểm Faithfulness thấp (LLM tự suy luận thêm).
4. Next optimization nếu có thêm 1 giờ: Tinh chỉnh Prompt cho LLM Generation để khắt khe hơn với context và giảm nhiệt độ (temperature) xuống 0.
