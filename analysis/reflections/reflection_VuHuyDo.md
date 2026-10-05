# Individual Reflection — Lab 18: Production RAG

**Học viên:** Vu Huy Do (bản nháp; xác nhận tên trước khi nộp)
**Khóa:** K4 - Track 3A
**Ngày:** 2026-10-05

> Bản nháp dựa trên triển khai và lần chạy trong workspace này. Cá nhân hóa phần action plan theo project thật của bạn trước khi nộp.

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng cosine similarity giữa embeddings; cấu hình threshold là 0.85. Có fallback khi không tải được model. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Tách parent 2048 ký tự và child 256 ký tự; pipeline chạy tạo 125 chunks từ 26 tài liệu. |
| BM25 + Dense fusion | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 bắt từ khóa sau khi segment tiếng Việt; dense search dùng bge-m3/Qdrant; RRF hợp nhất theo thứ hạng. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Chấm lại các ứng viên truy hồi và lấy top-3; fallback lexical dùng khi CrossEncoder không khả dụng. Chưa có benchmark latency đáng tin cậy. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Có đủ Faithfulness, Answer Relevancy, Context Precision, Context Recall. Gemini trả HTTP 429 do hết quota miễn phí, nên chưa có điểm hợp lệ để kết luận chất lượng. |
| Contextual enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | Combined mode gom summary, câu hỏi, context và metadata trong một request; fallback giữ nguyên nội dung khi thiếu key. |

## Phần 2: Khó khăn và cách xử lý

- **Lỗi API:** `This model models/gemini-2.5-flash is no longer available to new users.` Đổi model sang `gemini-3.8-flash` theo danh sách Google API. Request tiếp theo trả HTTP 429 vì hết quota free tier; cần chờ quota reset hoặc dùng provider khác còn quota rồi chạy lại.
- **Hạn chế công cụ:** `bash: rg: command not found`. Dùng `grep -R` để kiểm tra TODO thay thế.
- **Kiến thức cần củng cố:** API Qdrant `query_points()`, schema đầu vào của RAGAS và cách đọc từng metric để phân biệt lỗi retrieval với lỗi generation.
- **Kiểm chứng code:** lần chạy trước hoàn tất pipeline với fallback; sau đó 37 unit tests đã pass. Chưa thể chạy lại baseline/production có Gemini hoặc lấy điểm RAGAS vì quota đã hết.

## Phần 3: Action Plan cho project

### Project: Production RAG Pipeline (Lab 18)

#### Hiện trạng

- **Pipeline:** markdown/PDF text → hierarchical chunks → enrichment → BM25 + dense Qdrant + RRF → CrossEncoder → LLM → RAGAS.
- **Vấn đề đã xác nhận:** chưa có kết quả evaluation thật do quota Gemini free tier đã hết; PDF scan bị bỏ qua vì chưa có OCR.

#### Kế hoạch áp dụng

1. **Chunking:** giữ parent-child cho truy hồi child và trả parent làm ngữ cảnh; so sánh thêm structure-aware cho tài liệu Markdown.
2. **Search:** dùng hybrid BM25 + dense + RRF để kết hợp exact keyword và semantic matching.
3. **Reranking:** dùng `BAAI/bge-reranker-v2-m3` để xếp lại candidate; ghi latency trên cùng một bộ câu hỏi.
4. **Evaluation:** chạy đủ 20 câu với RAGAS, lưu baseline/production, đọc bottom-5 trước khi thay đổi pipeline.
5. **Enrichment:** dùng combined mode khi có API key; kiểm tra chi phí và retrieval trước/sau trên cùng test set.

#### Timeline

- **Tuần 1:** thay key đã lộ, bảo đảm còn quota, chạy lại baseline và production, lưu metric per-question.
- **Tuần 2:** phân tích bottom-5, thử một thay đổi mỗi lần (chunking, retrieval hoặc reranking), so sánh điểm và latency.