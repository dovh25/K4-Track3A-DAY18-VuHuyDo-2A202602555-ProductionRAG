# Failure Analysis — Lab 18: Production RAG

**Học viên:** Vu Huy Do (nháp, xác nhận lại tên trước khi nộp)
**Khóa:** K4 - Track 3A  
**Trạng thái:** Chưa có điểm RAGAS hợp lệ. Gemini API key có quyền truy cập model, nhưng free-tier quota cho `gemini-3.8-flash` đã hết (HTTP 429) trong lần kiểm tra ngày 2026-10-05.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | Chưa đo | Chưa đo | Chưa xác định |
| Answer Relevancy | Chưa đo | Chưa đo | Chưa xác định |
| Context Precision | Chưa đo | Chưa đo | Chưa xác định |
| Context Recall | Chưa đo | Chưa đo | Chưa xác định |

Report hiện ghi các metric bằng 0 để giữ schema, nhưng đây **không phải điểm chất lượng**. Lần chạy pipeline trước đó không có API key; lần thử Gemini sau đó xác nhận hết quota trước khi chạy evaluation. Pipeline đã xử lý 20 câu hỏi và 125 chunks từ 26 tài liệu ở lần chạy trước. Vì không có điểm theo từng câu từ một lần đánh giá hợp lệ, chưa thể xếp hạng hoặc kết luận bottom-5 một cách trung thực.

## Bottom-5 Failures

Chưa thể lập danh sách bottom-5. Sau khi thay key đã bị lộ và có quota khả dụng (hoặc dùng provider tương thích OpenAI khác), chạy `python main.py`, rồi dùng `reports/ragas_report.json` để điền câu hỏi, expected/got, metric thấp nhất, nguyên nhân và fix tương ứng. Không suy diễn câu trả lời sai từ các điểm 0 hiện tại.

## Diagnostic Tree hiện tại

1. **Output đúng?** Chưa đánh giá được bằng RAGAS; lần chạy pipeline trước dùng fallback do chưa có key.
2. **Context đúng và đủ?** Chưa có metric precision/recall hợp lệ để kết luận.
3. **Truy hồi và reranking có vấn đề không?** Cần kiểm tra từng context của các câu có điểm thấp sau khi chạy đánh giá thật.
4. **Nguyên nhân đã xác nhận:** Google API trả HTTP 429 vì free-tier request quota của model đã hết.
5. **Fix tiếp theo:** thu hồi key đã lộ, thay bằng key mới trong `.env`, chờ quota reset hoặc dùng provider có quota còn lại; sau đó chạy lại `python main.py`.

## Case Study

**Câu hỏi chọn phân tích:** Chưa chọn; cần report có điểm per-question trước.

**Nếu có thêm 1 giờ:** sau khi quota sẵn sàng, chạy đánh giá thật, kiểm tra top-5 lỗi theo từng metric, rồi ưu tiên sửa retrieval khi Context Recall thấp hoặc reranking khi Context Precision thấp.
