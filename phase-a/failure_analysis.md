# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: VPN công ty sử dụng giao  | reasoning | 0.40 | 0.83 | 0.50 | 1.00 | 0.68 | C2 |
| 2 | Kết hợp hai nội dung sau và trả lời ngắn gọn: Đánh giá hiệu suất nhân viên dựa trên tiêu c | multi_context | 1.00 | 0.85 | 0.33 | 1.00 | 0.80 | C1 |
| 3 | Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: Thuế GTGT phải nộp trong  | reasoning | 1.00 | 0.85 | 0.33 | 1.00 | 0.80 | C1 |
| 4 | Kết hợp hai nội dung sau và trả lời ngắn gọn: Tổng doanh thu hàng hóa bán ra của DHA Surfa | multi_context | 1.00 | 0.89 | 0.33 | 1.00 | 0.81 | C1 |
| 5 | Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: Thời gian thử việc là bao | reasoning | 1.00 | 0.75 | 0.50 | 1.00 | 0.81 | C1 |
| 6 | Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: Tài khoản bị khóa sau bao | reasoning | 0.50 | 0.80 | 1.00 | 1.00 | 0.83 | C2 |
| 7 | Tài khoản bị khóa sau bao nhiêu lần nhập sai mật khẩu? | simple | 0.50 | 0.83 | 1.00 | 1.00 | 0.83 | C2 |
| 8 | Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: Mật khẩu phải thay đổi sa | reasoning | 1.00 | 0.83 | 0.50 | 1.00 | 0.83 | C1 |
| 9 | Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: Tổng doanh thu hàng hóa b | reasoning | 1.00 | 0.89 | 0.50 | 1.00 | 0.85 | C1 |
| 10 | Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: Nhân viên có bao nhiêu ng | reasoning | 1.00 | 0.92 | 0.50 | 1.00 | 0.86 | C1 |

## Clusters Identified

### Cluster C1: Missing or weak retrieval context

**Pattern:** Questions have low context recall or context precision, usually because the first retrieved chunks do not contain enough evidence.

**Examples:**
- Kết hợp hai nội dung sau và trả lời ngắn gọn: Đánh giá hiệu suất nhân viên dựa trên tiêu chí nào? Đồng thời, So sánh giá trị hàng mua vào và bán ra của DHA Surfaces trong Q4/2024, bên nào lớn hơn và chênh lệch bao nhiêu?
- Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: Thuế GTGT phải nộp trong kỳ Q4/2024 của DHA Surfaces là bao nhiêu?
- Kết hợp hai nội dung sau và trả lời ngắn gọn: Tổng doanh thu hàng hóa bán ra của DHA Surfaces trong Q4/2024 là bao nhiêu? Đồng thời, Nhân viên có bao nhiêu ngân sách đào tạo mỗi năm?

**Root cause:** weakest metric often maps to `context_recall`.

**Proposed fix:** Increase top_k, add BM25+dense hybrid search, then rerank before generation.

### Cluster C2: Answer not grounded tightly enough

**Pattern:** The answer overlaps weakly with retrieved context or introduces extra text beyond the evidence.

**Examples:**
- Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: VPN công ty sử dụng giao thức gì?
- Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: Tài khoản bị khóa sau bao nhiêu lần nhập sai mật khẩu?
- Tài khoản bị khóa sau bao nhiêu lần nhập sai mật khẩu?

**Root cause:** weakest metric often maps to `faithfulness`.

**Proposed fix:** Use a stricter grounded prompt, temperature 0, and refuse when evidence is missing.

### Cluster C3: Multi-context synthesis failure

**Pattern:** Multi-context questions need facts from more than one document and the answer only covers part of the request.

**Examples:**
- Dựa trên tài liệu, giải thích ý nghĩa thực tế của thông tin sau: VPN công ty sử dụng giao thức gì?
- Kết hợp hai nội dung sau và trả lời ngắn gọn: Đánh giá hiệu suất nhân viên dựa trên tiêu chí nào? Đồng thời, So sánh giá trị hàng mua vào và bán ra của DHA Surfaces trong Q4/2024, bên nào lớn hơn và chênh lệch bao nhiêu?

**Root cause:** weakest metric often maps to `answer_relevancy`.

**Proposed fix:** Use query decomposition and retrieve per sub-question before final synthesis.
