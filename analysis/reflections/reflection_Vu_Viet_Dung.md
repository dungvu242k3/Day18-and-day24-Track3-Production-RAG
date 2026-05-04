# Individual Reflection — Lab 18

**Tên:** Vũ Việt Dũng  
**Module phụ trách:** M1, M2, M3, M4, M5 (Dự án cá nhân)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** Toàn bộ Pipeline từ OCR đến Evaluation (M1 - M5).
- **Các hàm/class chính đã viết:** 
    - `SemanticChunker`, `HierarchicalChunker` (M1)
    - `BM25Search`, `DenseSearch`, `HybridSearch` (M2)
    - `FlashrankReranker` (M3)
    - `evaluate_ragas`, `failure_analysis` (M4)
    - `ContextualEnricher` (M5)
- **Số tests pass:** 13/13 (M2: 5/5, M3: 5/5, M4: 3/3)

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Hybrid Search kết hợp thế mạnh của BM25 (keyword matching) và Dense Search (semantic) giúp xử lý tốt dữ liệu Tiếng Việt chuyên ngành. Cách thức đánh giá RAG định lượng bằng RAGAS.
- **Điều bất ngờ nhất:** Việc tăng độ phức tạp của Pipeline (Enrichment, Reranking) không phải lúc nào cũng tăng điểm RAGAS ngay lập tức. Trong một số trường hợp, Naive Baseline lại cho kết quả ổn định hơn do context ngắn gọn và sạch sẽ.
- **Kết nối với bài giảng (slide nào):** Slide Track 3 - Production RAG Pipeline: Modular RAG Architecture và Evaluation methods.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Thư viện RAGAS v0.2+ gặp lỗi tương thích nghiêm trọng với OpenAI Embeddings (`AttributeError: 'OpenAIEmbeddings' object has no attribute 'embed_query'`) và lỗi trả về `EvaluationResult` thay vì dict.
- **Cách giải quyết:** Đọc source code thư viện, triển khai `LangchainLLMWrapper` và `LangchainEmbeddingsWrapper`, đồng thời trích xuất dữ liệu trực tiếp từ DataFrame (`to_pandas()`) để đảm bảo tính ổn định của kết quả.
