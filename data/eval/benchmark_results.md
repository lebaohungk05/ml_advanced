# Báo cáo Đánh giá Hiệu năng Hệ thống (Benchmark Results)

> **Dự án:** Multimodal Fashion Search Engine  
> **Ngày cập nhật:** 30/08/2026  
> **Tác giả:** Đội ngũ phát triển (Hùng, Hiệp, Hưng, Hiếu)

---

## 1. Tổng quan Đánh giá & Quy mô Dữ liệu

Đánh giá được thực hiện trên tập dữ liệu kiểm thử thực tế của dự án nhằm so sánh độ chính xác và khả năng hiểu ngôn ngữ tiếng Việt của các hệ thống tìm kiếm đa mô thức (Multimodal Retrieval) và tìm kiếm từ khóa truyền thống.

* **Tổng số câu truy vấn tiếng Việt:** 155 truy vấn (bao gồm: truy vấn đơn thuộc tính, đa thuộc tính, phong cách/ngữ cảnh, và từ vựng bản địa).
* **Tổng số nhãn đánh giá độ liên quan:** **9.805 nhãn** (thang điểm: 0 = Không liên quan, 1 = Liên quan một phần, 2 = Liên quan chính xác).
* **Cơ chế đánh giá:** Pooling Top-20 từ các hệ thống nền tảng, kết hợp gán nhãn chéo độc lập (Double Blind Grading) và lọc tự động (Auto-zero).

---

## 2. Bảng Xếp hạng Hiệu năng các Hệ thống (Ranking Benchmark)

Được đo lường bằng các chỉ số chuẩn trong Hệ thống Tìm kiếm và Khai phá Thông tin (Information Retrieval):

| Hệ thống | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SigLIP 2 (Zero-shot)** | **0.5032** | **0.7677** | **0.8645** | **0.6145** | **0.5775** | **Hạng 1** — Vượt trội toàn diện, hiểu ngữ nghĩa tiếng Việt tự nhiên xuất sắc |
| **BM25 (Lexical Search)** | 0.1935 | 0.4516 | 0.5677 | 0.3048 | 0.2219 | **Hạng 2** — Hoạt động tốt với truy vấn có từ khóa khớp chính xác tiêu đề |
| **CLIP ViT-B/32 (Zero-shot)** | 0.1742 | 0.3742 | 0.4258 | 0.2546 | 0.1748 | **Hạng 3** — Bị giới hạn do tokenizer tiếng Anh không nắm bắt tốt tiếng Việt |
| **ViSigLIP-OT** | 0.0710 | 0.2065 | 0.2839 | 0.1324 | 0.0802 | **Hạng 4** — Kém hiệu quả trên miền dữ liệu thời trang Fashionpedia |

### Giải thích các chỉ số:
* **Recall@K:** Tỷ lệ tìm thấy ít nhất một sản phẩm liên quan trong Top-K kết quả trả về.
* **MRR (Mean Reciprocal Rank):** Vị trí nghịch đảo của kết quả liên quan đầu tiên (càng gần 1.0 thì kết quả đúng càng nằm ở vị trí đầu tiên).
* **nDCG@10 (Normalized Discounted Cumulative Gain):** Đánh giá chất lượng toàn diện của bảng xếp hạng Top-10 có tính đến mức độ liên quan nhiều mức ($0, 1, 2$) và vị trí giảm dần.

---

## 3. Đo lường Độ Tin Cậy Gán Nhãn (Inter-Annotator Agreement)

Nhằm đảm bảo tính khách quan khoa học, các sản phẩm trong pool được phân công cho ít nhất 2 thành viên chấm độc lập:

* **Số dòng được gán nhãn chéo ($\ge 2$ người chấm):** **3.427 dòng**
* **Hệ số tương quan Cohen's Kappa ($\kappa$):**
  - **Hiệp vs Hùng:** $\kappa = \mathbf{0.469}$ *(trên 2.178 dòng chung)*
  - **Hiệp vs Hưng:** $\kappa = \mathbf{0.442}$ *(trên 1.219 dòng chung)*
  - **Hùng vs Hưng:** $\kappa = \mathbf{0.452}$ *(trên 622 dòng chung)*

> 📌 **Kết luận thống kê:** Hệ số $\kappa \in [0.44, 0.47]$ đạt mức **Moderate Agreement** theo tiêu chuẩn Landis & Koch (1977), chứng minh bộ dữ liệu nhãn có độ nhất quán cao và đủ điều kiện làm Ground Truth.

---

## 4. Kết quả Huấn luyện Fine-tune SigLIP 2 + DoRA (`hung-run1`)

Đợt huấn luyện chính thức (Sprint 3) tinh chỉnh SigLIP 2 trên tập dữ liệu thời trang tiếng Việt:

* **Mô hình nền tảng (Backbone):** `google/siglip2-base-patch16-224`
* **Kỹ thuật thích ứng:** DoRA (Weight-Decomposed Low-Rank Adaptation)
  - LoRA Rank ($r$): 8
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `out_proj` (trên cả Text Tower và Image Tower)
  - Số tham số huấn luyện: **1.253.376** / 376.441.346 (**0,33%** tổng mô hình)
* **Khai thác mẫu âm khó (Hard Negative Mining):** Kích hoạt từ Epoch 5 (4 negatives / sample trong cùng danh mục)
* **Hàm mất mát:** Sigmoid Pairwise Loss (SigLIP loss)
* **Quy mô Batch:** Physical Batch = 32, Effective Batch = 256 (sử dụng GradCache 2 lượt forward)
* **Tổng số Epochs:** 10 / 10
* **Thời gian huấn luyện:** $11.084,7\text{ s}$ (~3 giờ 4 phút trên GPU RTX 5060 Laptop)
* **Validation Recall@5 (Self-query proxy):** **`0.2652`**
* **Checkpoint lưu trữ:**
  - `checkpoints/hung-run1/best/adapter_model.safetensors` (5.05 MB)
  - `checkpoints/hung-run1/best/adapter_config.json`
  - `checkpoints/hung-run1/best/README.md`

---

## 5. Phân tích Chuyên sâu & Định hướng Sprint Tiếp Theo

1. **Hiệu năng vượt trội của SigLIP 2:** Việc sử dụng bộ từ vựng đa ngữ rộng cùng kiến trúc Sigmoid loss giúp SigLIP 2 nắm bắt các cụm từ mô tả phong cách thời trang tiếng Việt (như *"áo khoác bomber nam màu be"*, *"quần ống suông cạp cao"*) tốt hơn hẳn CLIP truyền thống.
2. **Kế hoạch Sprint 4 (Hybrid Search & Reranking):**
   - Kết hợp điểm số Dense (SigLIP 2 + LoRA) và Sparse (BM25) qua thuật toán **Reciprocal Rank Fusion (RRF)** hoặc Weighted Sum.
   - Thử nghiệm Cross-Encoder Reranker trên Top-50 ứng viên để tối ưu hóa chỉ số nDCG@10 lên mức tối đa.
