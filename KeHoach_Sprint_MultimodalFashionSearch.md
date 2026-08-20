# Kế hoạch triển khai — 5 tuần (Multimodal Fashion Search)

> Rút gọn từ kế hoạch 12 tuần trong `DeCuong_MultimodalFashionSearch.md` xuống **5 sprint × 1 tuần**, nhờ bỏ khâu thu thập dữ liệu (dùng dataset công khai) và chia việc chạy song song giữa 4 người. Khối lượng công việc giữ nguyên như đề cương gốc — không cắt bớt phạm vi (150–200 truy vấn, top-20 pooling, đủ 7 trục ablation). Công nghệ cập nhật 08/2026: DoRA, ViSigLIP-OT (xem đề cương Mục 2.5). Theo dõi tiến độ trong `Project_Tracking.xlsx`.

## 0. Giả định

- **Dữ liệu:** một bộ dataset công khai duy nhất (FashionGen/DeepFashion/Fashionpedia — xem đề cương Mục 3.2) dùng cho cả train, index và catalog demo.
- **Công nghệ:** DoRA (`use_dora=True`, PEFT ≥ 0.18) là mặc định khi fine-tune; ViSigLIP-OT (0.2B, bản ngữ tiếng Việt) là backbone song song/baseline mới.

## 1. Phân công theo mảng

| Người | Mảng phụ trách | Nội dung chính |
|---|---|---|
| **Hùng** | Model &amp; Training | Pipeline SigLIP 2/ViSigLIP-OT/CLIP zero-shot, LoRA+DoRA + GradCache, fine-tune, ablation |
| **Hiếu** | Evaluation | Truy vấn, bộ chỉ số, chấm nhãn, phân tích lỗi, kiểm định thống kê |
| **Hiệp** | Data &amp; Systems/Demo | Dataset công khai, Qdrant, FastAPI, Streamlit, đo độ trễ, nhật ký thí nghiệm |
| **Hưng** | Data &amp; Report | Chốt dataset cụ thể, tổng hợp và hoàn thiện báo cáo cuối, theo dõi tiến độ chung |

## 2. Nhịp làm việc

- Mỗi sprint = 1 tuần, theo dõi task trong `Project_Tracking.xlsx`.
- **Giữa tuần:** họp nhanh 15–20 phút, mỗi người cập nhật tiến độ và nói luôn nếu đang vướng gì để cả nhóm cùng gỡ — không để đến cuối tuần mới phát hiện trễ.
- **Cuối tuần:** cả nhóm rà lại kết quả so với mục tiêu đã đặt cho sprint đó trước khi bắt đầu sprint mới. Nếu có phần chưa xong, bàn ngay cách xử lý (làm thêm 1–2 ngày, giảm phạm vi, hoặc dồn sang sprint sau) tuỳ mức độ ảnh hưởng.
- Kết quả chấm nhãn/đánh giá dùng chung được lưu lại (`relevance.json`, `runs.csv`...) để cuối kỳ viết báo cáo không phải lục lại từ đầu.

## 3. Tải công việc theo tuần

Hùng nặng nhất ở **tuần 3–4** (fine-tune + ablation). Hiếu nặng đều vì Evaluation là nút thắt của gần như mọi sprint (không có `queries.json`/`relevance.json` thì các phần khác không đánh giá được). Hiệp nặng ở đầu (dataset + hạ tầng) và cuối (demo). Hưng nặng dần từ tuần 2 vì vừa hỗ trợ chung khi cả nhóm cần rà lại số liệu/kết quả trước khi qua sprint mới, vừa viết báo cáo song song.

| | Tuần 1 | Tuần 2 | Tuần 3 | Tuần 4 | Tuần 5 |
|---|---|---|---|---|---|
| Hùng | Vừa | Vừa | **Nặng** | **Nặng** | Nhẹ |
| Hiếu | Vừa | **Nặng** | **Nặng** | **Nặng** | Vừa |
| Hiệp | **Nặng** | Vừa | Vừa | **Nặng** | **Nặng** |
| Hưng | Vừa | **Nặng** | **Nặng** | **Nặng** | **Nặng** |

## 4. Chi tiết 5 sprint

### Sprint 1 — Tuần 1: Setup + dataset công khai + khởi động truy vấn

**Mục tiêu:** repo chạy được · dataset đã chọn/tải/subsample đúng license · truy vấn có bản nháp.

- **Hùng:** cài `transformers`/`peft` (**≥0.18** cho DoRA)/GradCache; tải checkpoint SigLIP 2, ViSigLIP-OT, CLIP; smoke-test encode ảnh+text trên cả 2 backbone; viết ~50–65 truy vấn phần mình.
- **Hiếu:** thiết kế phân tầng truy vấn (đơn/đa thuộc tính, phong cách...); viết ~50–65 truy vấn phần mình; viết công thức Recall@K/MRR/nDCG@10.
- **Hiệp:** chọn dataset công khai (khuyến nghị FashionGen), tải, kiểm tra license, subsample ~20–50k, ánh xạ danh mục sang tiếng Việt; dựng repo skeleton, Qdrant docker, FastAPI skeleton, W&amp;B.
- **Hưng:** chốt dataset cụ thể (dựa trên đề xuất của Hiệp); viết ~50–65 truy vấn phần mình; viết khung báo cáo + Mục 1-2 (bối cảnh, phân tích phương pháp — viết được ngay, không cần chờ kết quả).

**Giao:** dataset công khai sẵn sàng · repo chạy được · `queries.json` đủ 150–200 truy vấn từ 3 người (Hùng/Hiếu/Hiệp) — Hưng viết thêm phần của mình nếu còn dư truy vấn cần đa dạng hơn.

---

### Sprint 2 — Tuần 2: Làm sạch + baseline zero-shot + hoàn tất truy vấn

**Mục tiêu:** dataset sạch + split sẵn sàng · `queries.json` hoàn tất · 5 baseline có số liệu.

- **Hùng:** baseline CLIP zero-shot; SigLIP 2 zero-shot (VI trực tiếp + dịch máy); **ViSigLIP-OT zero-shot** (bản ngữ, mới 08/2026).
- **Hiếu:** gộp/hoàn tất truy vấn thành `queries.json`; cài `evaluate.py`; baseline BM25.
- **Hiệp:** làm sạch (dedup, lọc danh mục), dùng lại split chính thức nếu có/tự chia 70-15-15, thống kê phân bố + gắn cờ danh mục 0 mẫu; index Qdrant; baseline ResNet-50+KNN.
- **Hưng:** viết Mục 3.1-3.2 báo cáo (kiến trúc tổng thể, dữ liệu) dựa trên số liệu thống kê của Hiệp.

**Giao:** dataset sạch + báo cáo thống kê · `queries.json` (150–200 truy vấn) · bảng kết quả 5 baseline.

---

### Sprint 3 — Tuần 3: Hệ thống đầu-cuối + chấm nhãn + bắt đầu fine-tune

**Mục tiêu:** hệ thống zero-shot đầu-cuối chạy được (2 backbone) · `relevance.json` + kappa · fine-tune vòng 1 đang chạy.

- **Hùng:** hoàn thiện pipeline đầu-cuối (SigLIP 2 + ViSigLIP-OT) qua Qdrant; cài LoRA+DoRA + sigmoid loss + GradCache; bắt đầu fine-tune vòng 1.
- **Hiếu:** điều phối chấm nhãn (**top-20** pooling, 2 người/kết quả trong nhóm 3 người Hùng/Hiếu/Hiệp, xoay vòng); tính Cohen's kappa, thống nhất các ca lệch.
- **Hiệp:** script hard-negative mining; khung log W&amp;B/`runs.csv`; tham gia chấm nhãn.
- **Hưng:** viết Mục 3.3-3.4 báo cáo (thiết kế mô hình, thiết kế đánh giá) dựa trên input của Hùng/Hiếu.

**Giao:** hệ thống zero-shot đầu-cuối · `relevance.json` + hệ số kappa · checkpoint fine-tune vòng 1 đang chạy.

---

### Sprint 4 — Tuần 4: Fine-tune + 7 trục ablation + latency + phân tích lỗi/thiên lệch

**Mục tiêu:** checkpoint tốt nhất đã chọn · bảng ablation · bảng latency 3 thành phần · 15–20 ca lỗi phân loại + ghi chú thiên lệch văn hoá.

- **Hùng:** hoàn tất fine-tune (DoRA + hard-negative); chọn checkpoint theo Recall@5. Chạy **7 trục ablation** theo thứ tự ưu tiên (dừng ở đâu ghi rõ trong báo cáo, không âm thầm bỏ qua):
  1. DoRA vs LoRA thường (rẻ nhất, câu hỏi mới nhất)
  2. Hard negative mining có/không
  3. SigLIP 2 vs ViSigLIP-OT (trả lời câu hỏi trung tâm của đồ án)
  4. LoRA rank ∈ {4,8,16}
  5. Batch hiệu dụng ∈ {32,128,256}
  6. Adapter ảnh-only / text-only / cả hai
  7. SigLIP 1 vs SigLIP 2
- **Hiếu:** đánh giá checkpoint vs 6 hệ thống zero-shot; chọn **15–20 ca lỗi tệ nhất**, phân loại nguyên nhân; chạy paired bootstrap test.
- **Hiệp:** re-index với checkpoint mới; đo latency 3 thành phần (p50/p95) cho tất cả hệ thống.
- **Hưng:** tổng hợp phân tích thiên lệch văn hoá (danh mục 0 mẫu) cùng Hiệp; viết Mục 5 báo cáo (kết quả, ablation, phân tích lỗi, thiên lệch) — phần nặng nhất của báo cáo vì hầu hết số liệu về đúng sprint này.

**Giao:** checkpoint tốt nhất + nhật ký thí nghiệm · bảng ablation (theo thứ tự ưu tiên trên) · bảng latency 3 thành phần · 15–20 ca lỗi + ghi chú thiên lệch văn hoá.

---

### Sprint 5 — Tuần 5: Demo + hoàn thiện báo cáo

**Mục tiêu:** demo chạy được đầu-cuối · báo cáo + slide hoàn chỉnh.

- **Hùng:** tổng hợp số liệu/kết quả mô hình-huấn luyện-ablation gửi Hưng ghép vào báo cáo; hỗ trợ tối ưu inference nếu demo chậm.
- **Hiếu:** tổng hợp số liệu đánh giá/kết quả/phân tích lỗi gửi Hưng; kiểm thử số liệu tái lập đúng như báo cáo.
- **Hiệp:** hoàn thiện FastAPI + Streamlit; tổng hợp mô tả kiến trúc hệ thống + latency; làm slide trình bày.
- **Hưng:** ghép Mục 1-2 (Sprint 1) + Mục 3.1-3.4 (Sprint 2-3) + Mục 5 (Sprint 4) đã viết sẵn, viết abstract/kết luận, format lại toàn bộ báo cáo. Vì đã viết trải đều 4 sprint trước, tuần này chỉ còn hoàn thiện.

**Giao:** demo Streamlit + API REST · báo cáo hoàn chỉnh + slide + buổi tập trình bày.

## 5. Rủi ro

| Rủi ro | Mức độ | Phương án dự phòng |
|---|---|---|
| Sprint 4 không đủ giờ máy để chạy hết 7 trục ablation | **Cao** | Đã có thứ tự ưu tiên (Mục 4, Sprint 4) — dừng ở đâu ghi rõ trong báo cáo, không cắt giấu |
| Dataset công khai thiếu ảnh trang phục truyền thống Việt Nam (áo dài...) | Cao *(đã lường trước)* | Loại khỏi Recall định lượng, chỉ báo cáo quan sát định tính (đề cương Mục 3.2/6.1) |
| Máy Hùng nghẽn ở tuần 3–4 | Trung bình | Giảm LoRA rank, ảnh 224→192, hoặc GPU cloud ngắn hạn cho riêng vòng fine-tune |
| Chấm nhãn Sprint 3 kéo dài dù có 3 người xoay vòng | Trung bình | Rút top-20 xuống top-10 nếu cả nhóm thấy không kịp |
| Một người chậm tiến độ ở sprint nào đó | Trung bình | Bàn ngay ở buổi giữa tuần, người khác hỗ trợ nếu còn dư giờ (ví dụ tuần đó đang "Nhẹ") |

## 6. Nếu thực sự chỉ có đúng 4 tuần

Gộp Sprint 4–5. Khi đó mới cắt scope: chỉ chạy 3 trục ablation đầu tiên trong danh sách ưu tiên (DoRA vs LoRA, hard-negative on/off, SigLIP2 vs ViSigLIP-OT), rút truy vấn xuống 100–120, rút phân tích lỗi xuống 10 ca, demo chạy qua script/notebook thay vì Streamlit hoàn chỉnh.
