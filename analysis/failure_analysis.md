# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Nhóm A
**Thành viên:** Vũ Việt Dũng (Làm cá nhân)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9917 | 0.7263 | -0.2654 |
| Answer Relevancy | 0.8135 | 0.6821 | -0.1314 |
| Context Precision | 0.8917 | 0.8553 | -0.0364 |
| Context Recall | 0.9300 | 0.8350 | -0.0950 |

## Bottom-5 Failures

### #1
- **Question:** Dữ liệu cá nhân cơ bản bao gồm những gì?
- **Expected:** Dữ liệu cá nhân cơ bản gồm: họ tên, ngày sinh, giới tính, nơi sinh, quốc tịch, hình ảnh, số điện thoại, CMND, CCCD, hộ chiếu, GPLX, biển số xe, mã số thuế, BHXH, BHYT, tình trạng hôn nhân, quan hệ gia đình, tài khoản số.
- **Worst metric:** Faithfulness (0.0)
- **Error Tree:** Output sai → Context đúng? (Đúng) → Query OK? (Đúng) → LLM Hallucination? (Đúng)
- **Root cause:** Context quá dài và chi tiết khiến LLM bỏ sót một số trường thông tin hoặc tự diễn giải lại theo ý hiểu thay vì trích xuất nguyên văn.
- **Suggested fix:** Sử dụng kĩ thuật Extract-then-Answer hoặc giảm số lượng chunk trả về.

### #2
- **Question:** Dữ liệu cá nhân nhạy cảm gồm những loại nào?
- **Worst metric:** Faithfulness (0.0)
- **Root cause:** LLM tự đưa thêm các ví dụ thực tế không có trong văn bản luật vào câu trả lời.
- **Suggested fix:** Thêm "Strict Context" instruction vào system prompt.

### #3
- **Question:** Những hành vi nào bị nghiêm cấm trong bảo vệ dữ liệu cá nhân?
- **Worst metric:** Context Precision (0.0)
- **Root cause:** Reranker không đẩy được chunk chứa đúng 5 hành vi cấm lên top 1, dẫn đến noise từ các điều khoản xử phạt.
- **Suggested fix:** Cải thiện logic Reranking hoặc dùng Hybrid search với trọng số BM25 cao hơn cho các từ khóa "nghiêm cấm".

### #4
- **Question:** Thuế GTGT phải nộp trong kỳ Q4/2024 của DHA Surfaces là bao nhiêu?
- **Worst metric:** Context Recall (0.2)
- **Root cause:** Chunking theo đoạn làm mất mối liên hệ giữa bảng biểu doanh thu và phần kết luận thuế ở cuối file MD.
- **Suggested fix:** Sử dụng Hierarchical Chunking với Parent context lớn hơn để giữ trọn vẹn thông tin bảng biểu.

### #5
- **Question:** Chuyển dữ liệu cá nhân ra nước ngoài được định nghĩa như thế nào?
- **Worst metric:** Answer Relevancy (0.5)
- **Root cause:** Câu trả lời quá ngắn gọn, thiếu các điều kiện về "không gian mạng" và "thiết bị điện tử".
- **Suggested fix:** Tăng max_tokens cho LLM Generation.

## Case Study (cho presentation)

**Question chọn phân tích:** "Dữ liệu cá nhân cơ bản bao gồm những gì?"

**Error Tree walkthrough:**
1. Output đúng? → Không hoàn toàn (thiếu 1 số trường như Biển số xe).
2. Context đúng? → Có, context chứa đầy đủ Điều 2 của Nghị định.
3. Query rewrite OK? → OK.
4. Fix ở bước: LLM Generation (Prompt engineering).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm query expansion/rewrite trước search.
- Tăng top_k cho hybrid search từ 10 lên 20.
- Tinh chỉnh Prompt để LLM chỉ được phép trích xuất (Extraction focus) thay vì tóm tắt.
