# ĐỀ CƯƠNG BÀI TẬP LỚN

## Hệ thống Tìm kiếm Sản phẩm Thời trang Đa mô thức cho Thương mại điện tử Tiếng Việt
### (Multimodal Fashion Product Search for Vietnamese E-commerce)

| | |
|---|---|
| **Nhóm thực hiện** | *(điền tên, MSSV các thành viên)* |
| **Giảng viên hướng dẫn** | *(điền)* |
| **Học phần** | *(điền)* |
| **Ngày trình đề cương** | *(điền)* |
| **Trạng thái** | Bản đề cương xin ý kiến nhận xét — chưa triển khai |

> **Ghi chú gửi giảng viên:** Đây là bản đề cương để xin nhận xét trước khi nhóm bắt đầu triển khai. Các phần được đánh dấu **[Cần ý kiến thầy/cô]** là những điểm nhóm còn băn khoăn và mong được góp ý. Danh sách câu hỏi tổng hợp nằm ở **Mục 7** cuối báo cáo.

---

## 1. Phát biểu bài toán

### 1.1. Bối cảnh

Trên các sàn thương mại điện tử Việt Nam (Shopee, Lazada, TikTok Shop), việc tìm kiếm sản phẩm thời trang hiện chủ yếu dựa trên **so khớp từ khoá** (keyword matching) với tên và mô tả sản phẩm. Cách này gặp ba hạn chế thực tế:

1. **Người mua không biết gọi tên sản phẩm.** Người dùng nhìn thấy một mẫu áo trên Instagram, muốn tìm mẫu tương tự, nhưng không biết nó gọi là "áo blazer oversize" hay "áo vest dáng rộng".
2. **Mô tả sản phẩm của người bán nhiễu nặng.** Tên sản phẩm trên sàn thường bị nhồi từ khoá (`ÁO SƠ MI NAM cao cấp ⚡FREESHIP⚡ hàng loại 1 xuất xịn TN01`), khiến so khớp từ khoá trả về kết quả kém liên quan.
3. **Không truy vấn được bằng hình ảnh một cách ngữ nghĩa.** Tính năng tìm bằng ảnh hiện có thường chỉ so khớp thị giác bề mặt, không hiểu được ý định ("tìm mẫu tương tự nhưng dài tay").

### 1.2. Phát biểu bài toán

Xây dựng hệ thống truy hồi (retrieval) cho phép người dùng tìm sản phẩm thời trang bằng **truy vấn văn bản tiếng Việt tự nhiên** hoặc bằng **ảnh**, trả về danh sách sản phẩm được xếp hạng theo độ tương đồng ngữ nghĩa.

**Định nghĩa hình thức:**

- Cho tập sản phẩm $\mathcal{P} = \{p_1, ..., p_N\}$, mỗi sản phẩm gồm ảnh $I_i$, tên/mô tả $T_i$ và metadata $M_i$ (giá, danh mục, shop).
- **Input:** truy vấn $q$ ở một trong hai dạng — văn bản tiếng Việt $q_t$, hoặc ảnh $q_i$.
- **Output:** danh sách $K$ sản phẩm xếp hạng giảm dần theo điểm tương đồng $s(q, p_j) \in [-1, 1]$.
- **Cơ chế:** học một không gian nhúng chung (joint embedding space) trong đó ảnh và văn bản mô tả cùng một sản phẩm có vector gần nhau.

### 1.3. Loại bài toán

Đây là bài toán **cross-modal retrieval / representation learning**, **không phải bài toán phân loại (classification)**. Điều này quyết định trực tiếp việc lựa chọn hàm mất mát (contrastive loss thay vì cross-entropy trên nhãn) và bộ chỉ số đánh giá (xem Mục 5.1 — nhóm **không** sử dụng Accuracy và giải thích lý do).

### 1.4. Yêu cầu và ràng buộc kỹ thuật

| Tiêu chí | Mục tiêu |
|---|---|
| Quy mô dữ liệu | 20.000 – 50.000 sản phẩm |
| Độ trễ truy vấn (end-to-end) | < 200 ms trên 1 GPU đơn |
| Phần cứng | 1 × NVIDIA RTX 4070 (12 GB VRAM) *(xem ghi chú 1.5)* |
| Ngôn ngữ truy vấn | Tiếng Việt (chính), tiếng Anh (phụ) |
| Triển khai | API REST + giao diện web demo chạy local |

*(Cập nhật 08/2026: dữ liệu lấy từ dataset công khai — xem Mục 3.2 — nên nhóm sẽ subsample về đúng khoảng 20.000–50.000 sản phẩm để giữ nguyên ràng buộc độ trễ/quy mô index ở trên, dù dataset gốc có thể lớn hơn nhiều.)*

### 1.5. Ghi chú về phần cứng

Nhóm cần chốt lại chính xác GPU sử dụng, vì **RTX 4070 bản desktop có 12 GB VRAM còn bản laptop chỉ có 8 GB**. Toàn bộ kế hoạch huấn luyện trong đề cương này được tính cho cấu hình **12 GB**. Nếu thực tế là bản laptop 8 GB, nhóm sẽ giảm kích thước ảnh đầu vào từ 224 xuống 192 và tăng số bước gradient accumulation tương ứng.

### 1.6. Tác động dự kiến

- **Với người mua:** giảm số lần truy vấn lại để tìm được sản phẩm mong muốn; đặc biệt hữu ích cho nhóm sản phẩm mà người dùng khó diễn đạt bằng lời.
- **Với người bán nhỏ:** sản phẩm được tìm thấy dựa trên nội dung ảnh thật thay vì phụ thuộc vào kỹ năng nhồi từ khoá SEO — giảm lợi thế không công bằng của các shop lớn có nhân sự làm SEO.
- **Ở phạm vi rộng hơn:** phần lớn mô hình vision-language hiện có được huấn luyện chủ yếu trên dữ liệu tiếng Anh. Việc đánh giá và tinh chỉnh chúng cho tiếng Việt trong một miền cụ thể (thời trang) là đóng góp có thể mở rộng sang các ngôn ngữ và miền ít tài nguyên khác.

---

## 2. Phân tích các phương pháp hiện có

### 2.1. Ba nhóm tiếp cận

**Nhóm A — Truy hồi dựa trên từ khoá (lexical retrieval)**
Biểu diễn văn bản bằng TF-IDF / BM25, so khớp từ khoá truy vấn với tên và mô tả sản phẩm.
*Ưu:* cực nhanh, không cần GPU, dễ giải thích, xử lý tốt tên riêng và mã sản phẩm.
*Nhược:* không hiểu đồng nghĩa ("áo croptop" vs "áo lửng"), hoàn toàn không dùng được thông tin ảnh, bị nhiễu nặng bởi từ khoá spam.

**Nhóm B — Truy hồi ảnh dựa trên đặc trưng CNN**
Trích xuất vector đặc trưng ảnh bằng CNN tiền huấn luyện (ResNet-50, EfficientNet), tìm láng giềng gần nhất bằng cosine similarity.
*Ưu:* đơn giản, ổn định, tìm sản phẩm giống nhau về thị giác khá tốt.
*Nhược:* **chỉ hỗ trợ ảnh-tìm-ảnh**, không nhận truy vấn văn bản; đặc trưng học từ ImageNet nên nhạy với nền và tư thế người mẫu hơn là với thuộc tính trang phục.

**Nhóm C — Mô hình vision-language contrastive**
CLIP, SigLIP, SigLIP 2 — huấn luyện đồng thời image encoder và text encoder để đưa hai mô thức về cùng không gian nhúng.
*Ưu:* hỗ trợ cả text→ảnh và ảnh→ảnh trong một mô hình duy nhất; khả năng zero-shot tốt; có thể tinh chỉnh rẻ bằng adapter.
*Nhược:* nặng hơn, cần GPU; chất lượng với tiếng Việt và từ vựng thời trang bản địa còn là câu hỏi mở — đây chính là khoảng trống mà đồ án nhắm tới.

### 2.2. So sánh chi tiết

| Phương pháp | Hỗ trợ text→ảnh | Hỗ trợ ảnh→ảnh | Hiểu ngữ nghĩa | Tiếng Việt | Chi phí tính toán | Cần fine-tune |
|---|---|---|---|---|---|---|
| BM25 | Có (lexical) | Không | Rất kém | Tốt (nếu tách từ đúng) | Rất thấp | Không |
| ResNet-50 + KNN | Không | Có | Trung bình | N/A | Thấp | Không |
| CLIP ViT-B/32 | Có | Có | Tốt | Kém (chỉ tiếng Anh) | Trung bình | Tuỳ chọn |
| SigLIP (base) | Có | Có | Tốt hơn CLIP | Kém | Trung bình | Tuỳ chọn |
| **SigLIP 2 (base)** | Có | Có | Tốt nhất trong nhóm | **Có hỗ trợ đa ngữ** | Trung bình | Tuỳ chọn |
| **ViSigLIP-OT (0.2B)** *— mới, xem 2.5* | Có | Có | Tốt, tối ưu cho ảnh-văn bản tiếng Việt | **Bản ngữ tiếng Việt** | Thấp (nhẹ hơn SigLIP 2 base) | Tuỳ chọn |
| Qwen3-VL-Embedding *(tham chiếu SOTA, xem 2.5)* | Có | Có | Rất tốt (SOTA MMEB-v2) | Tốt | Cao (kiến trúc LLM decoder) | Không khuyến nghị cho phạm vi này |
| Multimodal LLM embedding (nói chung) | Có | Có | Rất tốt | Tốt | **Rất cao** | Khó trên 12 GB |

### 2.3. Cơ sở lựa chọn

Nhóm chọn **SigLIP 2 (ViT-B) + LoRA adapter** làm mô hình chính, vì ba lý do kỹ thuật:

1. **Hàm mất mát sigmoid thay cho softmax contrastive.** CLIP dùng softmax trên toàn batch, nên chất lượng phụ thuộc mạnh vào batch size — điều bất lợi trên GPU 12 GB. SigLIP tách bài toán thành phân loại nhị phân độc lập trên từng cặp, hoạt động ổn hơn ở batch nhỏ.
2. **SigLIP 2 đã đa ngữ ngay từ tiền huấn luyện.** Đây là điểm quan trọng: nó khiến giả định "mô hình chỉ biết tiếng Anh" của các đề tài tương tự trở nên không còn đúng, và biến câu hỏi "SigLIP 2 zero-shot xử lý tiếng Việt tốt đến đâu?" thành một câu hỏi đo được — nhóm đưa nó thành một trong các thí nghiệm chính.
3. **Kiến trúc giữ nguyên so với SigLIP,** nên có thể hoán đổi trọng số để so sánh trực tiếp SigLIP vs SigLIP 2 mà không đổi code. *(Cập nhật 08/2026: cùng lý do này, nhóm có thể hoán đổi sang **ViSigLIP-OT** — xem Mục 2.5 — mà không đổi code, vì nó cũng giữ kiến trúc SigLIP.)*

Nhóm **không** chọn multimodal LLM embedding (Qwen3-VL-Embedding và tương tự) dù chúng mạnh hơn, vì không khả thi để tinh chỉnh trên 12 GB VRAM trong khuôn khổ một bài tập lớn — xem đánh giá cập nhật ở Mục 2.5.

### 2.4. Vấn đề đã biết: kiến trúc lai với text encoder tiếng Việt

Một phương án hay được đề xuất là dùng `bkai-foundation-models/vietnamese-bi-encoder` để xử lý truy vấn tiếng Việt. **Phương án này không thể triển khai trực tiếp**, vì đó là text encoder đơn mô thức, không gian nhúng của nó không đồng bộ với image encoder của SigLIP. Muốn dùng thì phải chưng cất kiến thức (distillation) theo kiểu Multilingual-CLIP: huấn luyện text encoder tiếng Việt bắt chước output của text encoder gốc.

**Cập nhật 08/2026 — vấn đề này giờ đã có lối ra không cần distillation:** nhóm phát hiện **ViSigLIP-OT / ViCLIP-OT** ([arXiv:2602.22678](https://arxiv.org/abs/2602.22678), công bố 2/2026) — mô hình vision-language nền tảng **đầu tiên huấn luyện riêng cho tiếng Việt**, giữ kiến trúc SigLIP nhưng thêm hàm mất mát Similarity-Graph Regularized Optimal Transport (SIGROT), chỉ 0.2 tỷ tham số, **trọng số công khai** tại [huggingface.co/collections/minhnguyent546/viclip-ot](https://huggingface.co/collections/minhnguyent546/viclip-ot). Đây đã là một image encoder + text encoder tiếng Việt **đồng bộ không gian nhúng ngay từ đầu** — đúng thứ mà hướng distillation ở trên muốn đạt được, nhưng không cần nhóm tự huấn luyện. Nhóm sẽ đưa nó vào làm hệ thống so sánh trực tiếp (Mục 5.2) thay vì chỉ ghi nhận là hướng mở rộng tương lai. Hướng distillation tổng quát (áp dụng cho miền/mô thức khác ngoài thời trang) vẫn còn giá trị cho đồ án tốt nghiệp, nhưng không còn là điều kiện tiên quyết cho bài tập lớn này.

**[Cần ý kiến thầy/cô]** Với phát hiện ViSigLIP-OT, nhóm có nên nâng nó lên làm một trong các backbone chính để fine-tune (song song SigLIP 2), hay chỉ dùng làm baseline zero-shot để đối chiếu?

### 2.5. Cập nhật công nghệ mới nhất (khảo sát 08/2026)

Trước khi triển khai, nhóm rà lại các lựa chọn ở Mục 2–5 theo tình hình công nghệ tính đến tháng 8/2026 để tránh dùng phương pháp đã lạc hậu. Tóm tắt:

| Hạng mục | Vẫn hợp lý — giữ nguyên | Nên bổ sung |
|---|---|---|
| Backbone chính | SigLIP 2 vẫn là mô hình mở mạnh nhất cho image–text similarity thuần tính đến giữa 2026 | Thêm **ViSigLIP-OT** (trên) làm ứng viên song song, ưu tiên vì bản ngữ tiếng Việt |
| PEFT | LoRA vẫn đúng hướng kỹ thuật | Bật **DoRA** (`use_dora=True`, PEFT ≥ 0.18) thay LoRA thường — xem Mục 3.3 |
| Batch lớn trên GPU nhỏ | GradCache vẫn là kỹ thuật chuẩn, chưa có phương án thay thế nổi bật hơn năm 2026 | — |
| Vector DB | Qdrant vẫn có latency thấp nhất (~p50 4ms) trong nhóm vector DB chuyên dụng | Có thể dùng Binary Quantization / GPU-accelerated indexing (bản 2026) nếu cần tối ưu thêm |
| Hard negative mining | Rule "cùng danh mục, khác màu/kiểu" vẫn dùng được, đủ cho phạm vi bài tập lớn | Tuỳ chọn nâng cấp attribute-guided theo AFMRL — xem Mục 3.3 |

**Tham chiếu SOTA, không đưa vào fine-tune — Qwen3-VL-Embedding:** bản 8B đạt 77.8 điểm MMEB-v2 (dẫn đầu benchmark tính đến đầu 2026, [arXiv:2601.04720](https://arxiv.org/abs/2601.04720)), kể cả bản 2B vẫn có thể fine-tune LoRA trong ~10GB VRAM. Tuy nhiên kiến trúc dựa trên LLM decoder (khác hẳn dual-encoder như SigLIP), suy luận nặng hơn và khó đảm bảo ràng buộc <200ms end-to-end — nhóm giữ quyết định ban đầu là **không** đưa vào pipeline fine-tune chính, chỉ chạy **zero-shot làm mốc tham chiếu SOTA** (Mục 5.2) để biết hệ thống của nhóm còn cách bao xa so với SOTA thực sự.

**Nguồn:**
- SigLIP 2 vẫn dẫn đầu về image–text similarity thuần, 6/2026: [spheron.network — Multimodal Embeddings on GPU Cloud](https://www.spheron.network/blog/multimodal-embedding-models-gpu-cloud-siglip2-jinaclip-cohere/)
- ViCLIP-OT / ViSigLIP-OT: [arXiv:2602.22678](https://arxiv.org/abs/2602.22678) · trọng số: [huggingface.co/collections/minhnguyent546/viclip-ot](https://huggingface.co/collections/minhnguyent546/viclip-ot)
- Qwen3-VL-Embedding & Reranker: [arXiv:2601.04720](https://arxiv.org/abs/2601.04720)
- DoRA: [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) · hỗ trợ chính thức trong PEFT 0.18 (6/2026): [github.com/huggingface/peft](https://github.com/huggingface/peft)
- AFMRL (ACL 2026 Findings): [arXiv:2604.20135](https://arxiv.org/abs/2604.20135)
- So sánh vector DB 2026: [kunalganglani.com — Milvus vs Qdrant 2026](https://www.kunalganglani.com/blog/milvus-vs-qdrant)

---

## 3. Thiết kế giải pháp

### 3.1. Kiến trúc tổng thể

```
                        ┌─────────────────────────────────┐
   TRUY VẤN            │      GIAI ĐOẠN OFFLINE          │
                        │  (index toàn bộ catalog 1 lần)  │
 ┌──────────────┐       │                                 │
 │ Text tiếng   │       │  Ảnh sản phẩm (N ≈ 30k)         │
 │ Việt         │       │            │                    │
 └──────┬───────┘       │            ▼                    │
        │               │   SigLIP 2 Image Encoder        │
        ▼               │   (frozen + LoRA)               │
  Text Encoder          │            │                    │
  (SigLIP 2 +           │            ▼                    │
   LoRA adapter)        │   Vector 768-d (L2 normalized)  │
        │               │            │                    │
        │               │            ▼                    │
        │               │   ┌──────────────────────┐      │
        │               │   │   Qdrant (HNSW)      │      │
        │               │   │  vector + payload    │      │
        └───────────────┼──▶│  {giá, danh mục,     │      │
   Vector 768-d         │   │   shop, rating}      │      │
                        │   └──────────┬───────────┘      │
 ┌──────────────┐       │              │                  │
 │ Ảnh truy vấn │──────▶│   Image Encoder (dùng lại)      │
 └──────────────┘       └──────────────┼──────────────────┘
                                       ▼
                          Cosine similarity + filter metadata
                                       ▼
                            Top-K sản phẩm + điểm tương đồng
```

Điểm thiết kế đáng chú ý: cả hai loại truy vấn (text và ảnh) đều được đưa về **cùng một không gian 768 chiều**, nên hệ thống chỉ cần **một index Qdrant duy nhất** cho mọi loại truy vấn.

### 3.2. Pipeline dữ liệu

**Cập nhật 08/2026 — quyết định mới về nguồn dữ liệu:** để không tốn thời gian thu thập và xử lý pháp lý, nhóm quyết định **toàn bộ dữ liệu — cả huấn luyện và đánh giá — lấy từ bộ dữ liệu công khai** có giấy phép nghiên cứu rõ ràng, bỏ hoàn toàn kế hoạch crawl 20.000–30.000 sản phẩm từ sàn TMĐT Việt Nam đã nêu trong bản đề cương gốc. Một số lựa chọn phù hợp (nhóm tự chốt bộ cụ thể sau khi kiểm tra license, không nằm trong phạm vi đề cương này):

| Bộ dữ liệu | Quy mô | Vì sao phù hợp |
|---|---|---|
| **FashionGen** | ~260.500 cặp ảnh–văn bản | Có mô tả tự nhiên đi kèm ảnh (không chỉ nhãn thuộc tính) — phù hợp trực tiếp cho contrastive learning kiểu SigLIP |
| **DeepFashion** | ~800.000 ảnh | Quy mô lớn, nhiều góc chụp/tư thế — tốt để học đặc trưng thị giác đa dạng |
| **Fashionpedia** | ~48.000 ảnh | Có nhãn thuộc tính chi tiết (segmentation + attribute) — hữu ích để xây taxonomy phân tích lỗi ở Mục 5.4 |
| Fashion-IQ | ~77.700 ảnh | Chỉ cần nếu muốn thử composed retrieval sơ bộ (Mục 6.4) — không bắt buộc |

Vì tên/mô tả gốc trong các bộ này là tiếng Anh, **150–200 truy vấn tiếng Việt tự soạn (Mục 3.4) vẫn giữ nguyên vai trò trung tâm của việc đánh giá** — người chấm nhãn nhìn ảnh sản phẩm để đánh giá độ liên quan, không phụ thuộc caption gốc, nên dữ liệu train tiếng Anh không ảnh hưởng đến tính hợp lệ của việc đánh giá bằng truy vấn tiếng Việt tự nhiên.

**Hệ quả quan trọng cần lường trước:** vì dữ liệu công khai phổ biến hiện nay chủ yếu chụp trang phục phương Tây, **catalog gần như chắc chắn không có (hoặc rất hiếm) ảnh áo dài, áo bà ba** — khác với dự đoán "ít mẫu" trong bản gốc, đây có thể là "không mẫu nào". Điều này ảnh hưởng trực tiếp đến khả năng đo **thiên lệch văn hoá** đã đặt ra ở Mục 6.3: nhóm sẽ cần loại các truy vấn về trang phục truyền thống Việt Nam khỏi Recall theo danh mục (vì không có đáp án đúng để tìm), và chỉ báo cáo quan sát định tính (ví dụ mô hình trả về gì khi không có kết quả đúng) thay vì Recall định lượng cho nhóm danh mục này — xem cập nhật ở Mục 6.1/6.3.

**Không còn áp dụng so với bản gốc:** toàn bộ phần thu thập dữ liệu TMĐT Việt Nam (crawl tốc độ thấp, tôn trọng `robots.txt`, không phân phối lại ảnh gốc...) đã bị loại khỏi phạm vi bài tập lớn. Câu hỏi xin ý kiến giảng viên về quy định thu thập dữ liệu web (Mục 7, câu 2 bản gốc) theo đó cũng không còn cần thiết. Nhóm ghi nhận việc đo trên dữ liệu TMĐT Việt Nam thật vẫn là hướng mở rộng hợp lệ cho đồ án tốt nghiệp.

**Làm sạch dữ liệu:** vì dữ liệu công khai đã được các nhóm nghiên cứu xử lý từ trước, khối lượng làm sạch giảm nhiều so với dữ liệu crawl thô ban đầu:

| Vấn đề | Cách xử lý |
|---|---|
| Ảnh lỗi, ảnh trắng, ảnh < 100px | Loại bỏ (hiếm gặp với dataset công khai, vẫn kiểm tra lại) |
| Ảnh trùng lặp | Phát hiện bằng perceptual hash, giữ 1 bản |
| Danh mục ngoài phạm vi thời trang thường ngày (nếu dataset có phạm vi rộng hơn cần) | Lọc theo danh mục liên quan |
| Mô tả quá ngắn (< 3 từ) hoặc thiếu ảnh | Loại bỏ |
| Chuẩn hoá Unicode NFC, loại emoji/cụm quảng cáo | **Chỉ áp dụng cho 150–200 truy vấn tiếng Việt do nhóm tự soạn** (Mục 3.4) — không áp dụng cho caption gốc tiếng Anh của dataset |

**Chia tập dữ liệu — điểm cần đặc biệt cẩn thận:** ưu tiên **dùng lại split chính thức theo sản phẩm/item id** nếu dataset đã cung cấp sẵn; nếu không, tự chia **theo sản phẩm (product-level split)**, không chia theo ảnh — vì một sản phẩm thường có nhiều ảnh gần giống nhau, nếu chia theo ảnh thì ảnh của cùng một sản phẩm sẽ xuất hiện ở cả tập huấn luyện và tập kiểm tra → rò rỉ dữ liệu (data leakage) và kết quả bị thổi phồng. Tỉ lệ dự kiến khi tự chia: train 70% / val 15% / test 15%.

**Kiểm tra mất cân bằng:** nhóm sẽ thống kê phân bố theo danh mục và báo cáo trong báo cáo cuối. Dự đoán lệch mạnh theo hướng trang phục phương Tây phổ biến (áo thun, áo sơ mi, quần jean chiếm phần lớn); các danh mục trang phục truyền thống Việt Nam có thể **bằng 0** như đã nêu trên. Cách xử lý: báo cáo Recall tách theo danh mục thay vì chỉ báo cáo số trung bình, và loại rõ các danh mục có 0 mẫu trong catalog khỏi phần Recall định lượng.

### 3.3. Kiến trúc mô hình và chiến lược huấn luyện

**Cấu hình adapter:**

- Backbone: `google/siglip2-base-patch16-224` — **đóng băng toàn bộ**.
- LoRA adapter chèn vào các lớp projection của attention (`q_proj`, `k_proj`, `v_proj`, `out_proj`) ở **cả** image encoder và text encoder.
- Rank $r = 8$ (mặc định), $\alpha = 16$, dropout 0.1.
- **Cập nhật 08/2026:** bật **DoRA** (`use_dora=True` trong `LoraConfig`, hỗ trợ chính thức từ PEFT 0.18) thay cho LoRA thường — cùng chi phí tham số nhưng cải thiện nhất quán hơn theo các so sánh gần đây trên cả tác vụ ngôn ngữ và thị giác ([arXiv:2402.09353](https://arxiv.org/abs/2402.09353)). Nhóm đưa "DoRA vs LoRA thường" vào ablation (Mục 5.3) để tự kiểm chứng trên bài toán của mình, không chỉ tin theo paper gốc.
- Số tham số huấn luyện dự kiến: khoảng 1–2% tổng số tham số → báo cáo con số chính xác trong báo cáo cuối.

**Hàm mất mát:** sigmoid pairwise loss theo đúng công thức của SigLIP, có bổ sung **hard negative mining** ở giai đoạn 2 của huấn luyện: với mỗi sản phẩm, lấy các sản phẩm cùng danh mục nhưng khác màu/khác kiểu làm mẫu âm khó. *(Tuỳ chọn nâng cấp 08/2026 nếu còn thời gian: định hướng mẫu âm theo thuộc tính trích xuất tự động — màu/chất liệu/pattern — kiểu AFMRL ([arXiv:2604.20135](https://arxiv.org/abs/2604.20135)) thay vì chỉ theo danh mục.)*

**Vấn đề batch size và giải pháp:** học tương phản (contrastive learning) cần batch lớn để có đủ mẫu âm, nhưng 12 GB VRAM chỉ cho phép batch khoảng 32–64. Giải pháp là **gradient caching (GradCache)** — kỹ thuật tính gradient theo hai lượt, cho phép mô phỏng batch hiệu dụng 256–512 trên GPU nhỏ. Nhóm coi việc đo ảnh hưởng của batch hiệu dụng lên chất lượng là một trong các thí nghiệm ablation chính (Mục 5.3).

**Siêu tham số dự kiến:**

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 (adapter), cosine decay, warmup 500 bước |
| Weight decay | 0.01 |
| Số epoch | 10, có early stopping |
| Batch vật lý / hiệu dụng | 32 / 256 (qua GradCache) |
| Precision | bf16 mixed precision |
| Tiêu chí chọn checkpoint | Recall@5 trên tập validation |
| Seed | Cố định (42), báo cáo trên 3 seed nếu kịp thời gian |

### 3.4. Thiết kế đánh giá — phần nhóm coi là quan trọng nhất

Đây là điểm nhóm cho rằng dễ mắc sai lầm nhất, nên trình bày riêng.

**Cách làm sai (nhóm chủ động tránh):** lấy tên sản phẩm gốc làm truy vấn và coi chính sản phẩm đó là đáp án đúng duy nhất. Cách này cho ra chỉ số Recall rất cao nhưng vô nghĩa, vì nó chỉ đo khả năng "tìm lại đúng caption của chính mình", không phản ánh cách người dùng thật gõ truy vấn.

**Cách làm của nhóm:**

1. Mỗi thành viên tự viết 40–50 truy vấn theo phong cách người mua thật (`"áo khoác bomber nam màu be"`, `"váy hai dây đi biển"`, `"quần baggy jean rộng ống"`), tổng **khoảng 150–200 truy vấn**. *(Cập nhật 08/2026: lịch rút gọn còn 5 tuần — xem `KeHoach_Sprint_MultimodalFashionSearch.md` — nhưng khối lượng này được giữ nguyên, không cắt giảm; việc nén lịch đạt được bằng cách bỏ khâu thu thập dữ liệu và chạy song song 4 người, không phải bằng cách giảm phạm vi công việc.)*
2. Xây tập truy vấn có phân tầng: truy vấn đơn thuộc tính, truy vấn nhiều thuộc tính, truy vấn dùng từ vựng thời trang bản địa/mượn tiếng Anh, và truy vấn theo phong cách ("phong cách Hàn Quốc", "công sở").
3. Chạy **tất cả** các hệ thống cần so sánh, gộp top-20 của mỗi hệ thống thành một danh sách chung, **xáo trộn và ẩn nguồn** (blind pooling).
4. Mỗi kết quả được **hai thành viên** chấm độc lập theo thang 3 mức: 0 = không liên quan, 1 = liên quan một phần, 2 = đúng ý. Báo cáo **độ đồng thuận giữa người chấm (Cohen's kappa)**; các trường hợp lệch sẽ được cả nhóm thống nhất lại.

Quy trình này cho phép dùng nDCG (chỉ số cần nhãn nhiều mức) và tránh vấn đề mẫu âm giả (false negative) — vấn đề mà chính các benchmark quốc tế như CIRCO được tạo ra để giải quyết.

**[Cần ý kiến thầy/cô]** Với khối lượng bài tập lớn, 150–200 truy vấn có được xem là đủ, hay nhóm nên giảm xuống để dồn thời gian cho phần khác?

---

## 4. Kế hoạch triển khai và công nghệ sử dụng

### 4.1. Bảng công nghệ đầy đủ kèm liên kết

**Mô hình (Hugging Face Hub)**

| Thành phần | Định danh | Vai trò |
|---|---|---|
| Mô hình chính | [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) | Backbone vision-language, đa ngữ |
| **Mô hình chính (song song)** *— mới, 08/2026* | [`minhnguyent546/ViSigLIP-OT`](https://huggingface.co/collections/minhnguyent546/viclip-ot) | Vision-language bản ngữ tiếng Việt, kiến trúc SigLIP + SIGROT, 0.2B tham số |
| Mô hình so sánh | [`google/siglip-base-patch16-256`](https://huggingface.co/google/siglip-base-patch16-256) | SigLIP bản 1, để đo mức cải thiện của bản 2 |
| Mô hình so sánh | [`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) | CLIP truyền thống, mốc tham chiếu |
| Mô hình tham chiếu SOTA *— mới, 08/2026* | [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | Zero-shot only — mốc SOTA MMEB-v2, không fine-tune (Mục 2.5) |
| Baseline ảnh | [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50) | Trích xuất đặc trưng cho baseline KNN |
| ~~Hướng mở rộng~~ *(đã thay bằng ViSigLIP-OT ở trên)* | [`bkai-foundation-models/vietnamese-bi-encoder`](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) | Text encoder tiếng Việt đơn mô thức — chỉ còn tham khảo, không cần distillation nữa (Mục 2.4) |

**Thư viện và framework (GitHub)**

| Hạng mục | Repo | Mục đích sử dụng |
|---|---|---|
| Fine-tuning hiệu quả tham số | [huggingface/peft](https://github.com/huggingface/peft) | Cài đặt LoRA adapter — dùng bản **≥ 0.18** (6/2026) để có `use_dora=True` (Mục 3.3) |
| Thư viện mô hình | [huggingface/transformers](https://github.com/huggingface/transformers) | Nạp SigLIP 2, tokenizer, processor |
| Vector database | [qdrant/qdrant](https://github.com/qdrant/qdrant) | Index HNSW + lọc theo metadata; bản 2026 có thêm Binary Quantization và GPU-accelerated indexing nếu cần tối ưu thêm |
| Vector DB thay thế | [chroma-core/chroma](https://github.com/chroma-core/chroma) | Phương án dự phòng nếu Qdrant gặp vấn đề |
| Contrastive batch lớn | [luyug/GradCache](https://github.com/luyug/GradCache) | Mô phỏng batch 256+ trên 12 GB VRAM |
| CLIP mã nguồn mở | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | Tham chiếu cài đặt loss và vòng huấn luyện |
| Tokenizer tiếng Việt | [VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT) | Tham chiếu chuẩn hoá và tách từ tiếng Việt |
| Backend API | [fastapi/fastapi](https://github.com/fastapi/fastapi) | REST API cho tìm kiếm |
| Giao diện demo | [streamlit/streamlit](https://github.com/streamlit/streamlit) | UI demo kiểu sàn TMĐT |
| Theo dõi thí nghiệm | [wandb/wandb](https://github.com/wandb/wandb) | Nhật ký thí nghiệm, biểu đồ loss |
| ANN thay thế | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | Đối chiếu tốc độ với Qdrant |

**Dữ liệu tham chiếu** *(cập nhật 08/2026 — nguồn dữ liệu chính, xem Mục 3.2; nhóm tự chốt bộ cụ thể sau khi kiểm tra license)*

| Bộ dữ liệu | Nguồn |
|---|---|
| DeepFashion | [mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) |
| FashionGen | *(nhóm sẽ xác nhận nguồn phân phối chính thức và giấy phép trước khi dùng)* |
| Fashionpedia | *(nhóm sẽ xác nhận nguồn phân phối chính thức và giấy phép trước khi dùng)* |
| FashionIQ | [github.com/XiaoxiaoGuo/fashion-iq](https://github.com/XiaoxiaoGuo/fashion-iq) |

**Tài liệu tham khảo chính**

| Công trình | Định danh |
|---|---|
| SigLIP 2: Multilingual Vision-Language Encoders (2025) | [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) |
| CLIP: Learning Transferable Visual Models (2021) | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| LoRA: Low-Rank Adaptation of LLMs (2021) | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| FineCIR: Fine-Grained Modification Semantics for CIR (2025) | [arXiv:2503.21309](https://arxiv.org/abs/2503.21309) |
| Qwen3-VL-Embedding & Reranker (2026) | [arXiv:2601.04720](https://arxiv.org/abs/2601.04720) |
| **ViCLIP-OT: Foundation VL Model for Vietnamese (2026)** *— mới* | [arXiv:2602.22678](https://arxiv.org/abs/2602.22678) |
| **DoRA: Weight-Decomposed Low-Rank Adaptation (2024)** *— mới* | [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) |
| **AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce (2026)** *— mới* | [arXiv:2604.20135](https://arxiv.org/abs/2604.20135) |

> *Nhóm sẽ kiểm tra lại toàn bộ liên kết và bổ sung đầy đủ danh mục tài liệu tham khảo theo định dạng chuẩn (IEEE hoặc APA — xin thầy/cô cho biết định dạng yêu cầu) trong báo cáo cuối.*

### 4.2. Cấu trúc mã nguồn dự kiến

```
fashion-multimodal-search/
├── data/
│   ├── raw/                  # dataset công khai đã tải, không commit
│   ├── processed/            # sau khi làm sạch (parquet)
│   └── eval/
│       ├── queries.json      # 150-200 truy vấn tự soạn
│       └── relevance.json    # nhãn 0/1/2 do nhóm chấm
├── src/
│   ├── data_prep/             # tải + làm sạch dataset công khai (không còn crawl/)
│   ├── dataset.py            # Dataset, collate, chia theo sản phẩm
│   ├── model.py              # SigLIP 2 / ViSigLIP-OT + cấu hình LoRA+DoRA
│   ├── loss.py               # sigmoid loss, hard negative mining
│   ├── train.py              # vòng huấn luyện + GradCache
│   ├── index.py              # encode toàn catalog, đẩy vào Qdrant
│   ├── search.py             # logic truy vấn + lọc metadata
│   ├── evaluate.py           # Recall@K, MRR, nDCG@10, latency
│   └── api.py                # FastAPI endpoints
├── app/
│   └── streamlit_app.py      # giao diện demo
├── notebooks/                # phân tích dữ liệu, phân tích lỗi
├── experiments/
│   └── runs.csv              # NHẬT KÝ THÍ NGHIỆM (xem 4.4)
├── configs/                  # file cấu hình YAML cho từng lần chạy
├── requirements.txt
└── README.md
```

### 4.3. Tiền xử lý

- **Ảnh:** resize về 224×224 (giữ tỉ lệ, pad viền), chuẩn hoá theo tham số của SigLIP 2 processor. Augmentation khi huấn luyện: random crop, horizontal flip, color jitter nhẹ (**không** dùng color jitter mạnh vì màu sắc là thuộc tính quan trọng cần giữ nguyên trong bài toán này).
- **Văn bản:** **giữ nguyên dấu tiếng Việt** (không bỏ dấu, vì tokenizer đa ngữ của SigLIP 2 xử lý được và bỏ dấu sẽ gây nhập nhằng nghiêm trọng). Chuẩn hoá Unicode NFC, chuyển chữ thường, loại emoji và cụm quảng cáo, giới hạn 64 token.

### 4.4. Nhật ký thí nghiệm

Nhóm sẽ ghi lại **mọi** lần chạy huấn luyện vào `experiments/runs.csv` và đồng bộ với W&B, gồm các cột: `run_id`, ngày, người chạy, commit hash, file config, LoRA rank, learning rate, batch hiệu dụng, số epoch, Recall@1/5/10 trên val, thời gian chạy, ghi chú. Việc này được thiết lập **ngay từ lần chạy đầu tiên** để tránh phải truy hồi số liệu từ log terminal khi viết báo cáo.

### 4.5. Timeline

> **Cập nhật 08/2026:** lịch chính thức của nhóm là **5 tuần**, không phải 12 tuần — nhưng **khối lượng công việc không đổi so với bản gốc**. Nén được lịch là nhờ (1) bỏ hẳn khâu thu thập dữ liệu TMĐT Việt Nam (dùng dataset công khai, Mục 3.2) — vốn chiếm 2/12 tuần gốc — và (2) 4 người chạy nhiều track song song mỗi tuần thay vì mỗi tuần chỉ làm một nội dung tuần tự như bảng 12 tuần ban đầu. Chi tiết công việc từng người từng sprint nằm ở file riêng `KeHoach_Sprint_MultimodalFashionSearch.md`, được theo dõi và kiểm soát chất lượng qua `Project_Tracking.xlsx`.

| Tuần | Nội dung (dồn từ 12 tuần gốc) | Sản phẩm giao |
|---|---|---|
| 1 | Setup môi trường + chọn/tải dataset công khai + khảo sát tài liệu *(gộp tuần 1–2 gốc)* | Repo chạy được, dataset công khai đã chọn &amp; subsample |
| 2 | Làm sạch + chia tập + soạn 150–200 truy vấn + baseline BM25/ResNet-50+KNN/CLIP/SigLIP 2 ×2/ViSigLIP-OT *(gộp tuần 2–4 gốc)* | `queries.json` (150–200 truy vấn), bảng kết quả 5 baseline |
| 3 | Pipeline zero-shot đầu-cuối + chấm nhãn cả nhóm + bắt đầu fine-tune *(gộp tuần 5–7 gốc)* | Hệ thống đầu-cuối, `relevance.json` + hệ số kappa, checkpoint vòng 1 |
| 4 | Hoàn tất fine-tune + 7 trục ablation (Mục 5.3) + đo độ trễ + phân tích lỗi/thiên lệch *(gộp tuần 8–10 gốc)* | Checkpoint tốt nhất, bảng ablation đầy đủ, bảng latency, 15–20 ca lỗi đã phân loại |
| 5 | Hoàn thiện demo + viết báo cáo, slide *(gộp tuần 11–12 gốc)* | Demo Streamlit + API REST, báo cáo + slide |

Mỗi tuần kết thúc bằng một buổi rà lại kết quả so với mục tiêu đã đặt trước khi cả nhóm chuyển sang tuần kế — chi tiết nhịp làm việc ở `KeHoach_Sprint_MultimodalFashionSearch.md` Mục 2.

### 4.6. Phân công

**Cập nhật 08/2026:** phân công dưới đây được rà soát lại cho gọn hơn, tận dụng việc bỏ khâu thu thập dữ liệu để dồn nhân lực vào các phần còn lại.

| Thành viên | Mảng phụ trách | Phụ trách chính | Phụ trách phụ |
|---|---|---|---|
| **Hưng** | Data &amp; Report | Chốt dataset cụ thể, tổng hợp và hoàn thiện báo cáo cuối, theo dõi tiến độ chung (`Project_Tracking.xlsx`) | Viết truy vấn, chấm nhãn |
| **Hùng** | Model &amp; Training | Mô hình, fine-tune, ablation | Tối ưu inference cho demo |
| **Hiếu** | Evaluation | Truy vấn, bộ chỉ số, chấm nhãn, phân tích lỗi | Kiểm định thống kê |
| **Hiệp** | Data + Systems/API/Demo | Chuẩn bị dữ liệu (tải/làm sạch/split), Qdrant, API, demo, đo độ trễ | Nhật ký thí nghiệm |

---

## 5. Kế hoạch đánh giá

### 5.1. Bộ chỉ số và lý do lựa chọn

| Chỉ số | Ý nghĩa | Lý do dùng |
|---|---|---|
| **Recall@1 / @5 / @10** | Tỉ lệ có ít nhất một sản phẩm đúng trong top K | Phản ánh trực tiếp trải nghiệm người dùng (chỉ xem 1–2 hàng đầu) |
| **MRR** | Nghịch đảo thứ hạng của kết quả đúng đầu tiên | Thưởng cho việc đưa kết quả đúng lên cao |
| **nDCG@10** | Chất lượng xếp hạng có tính đến mức độ liên quan | Cần thiết vì nhãn của nhóm có 3 mức (0/1/2) |
| **Encode latency (ms)** | Thời gian nhúng truy vấn trên GPU | Thành phần chi phối độ trễ thực tế |
| **ANN search latency (ms)** | Thời gian tìm kiếm trong Qdrant | Đo riêng để thấy đúng nút cổ chai |
| **End-to-end latency (ms)** | Tổng thời gian từ request đến response | Con số duy nhất có ý nghĩa với người dùng |

**Lý do không dùng Accuracy:** đây là bài toán xếp hạng, không phải phân loại. Không tồn tại khái niệm "dự đoán đúng/sai" cho một truy vấn, vì output là một danh sách có thứ tự và thường có nhiều sản phẩm đều liên quan ở mức độ khác nhau. Accuracy sẽ bỏ mất hoàn toàn thông tin về thứ hạng.

**Về việc báo cáo độ trễ một cách trung thực:** nhóm sẽ **không** báo cáo một con số kiểu "truy vấn trong 12 ms". Tìm kiếm HNSW trên 30.000 vector thực tế chỉ mất vài mili-giây, nhưng riêng bước nhúng truy vấn trên GPU đã tốn khoảng 10–40 ms, chưa kể tiền xử lý ảnh và overhead HTTP. Nhóm sẽ báo cáo phân rã ba thành phần kèm phân vị p50/p95.

### 5.2. Các hệ thống đưa vào so sánh

| # | Hệ thống | Vai trò |
|---|---|---|
| 1 | BM25 trên tên + mô tả | Baseline nhánh text |
| 2 | ResNet-50 + KNN cosine | Baseline nhánh ảnh |
| 3 | CLIP ViT-B/32 zero-shot | Mốc tham chiếu quốc tế |
| 4 | SigLIP 2 zero-shot, truy vấn tiếng Việt trực tiếp | Đo năng lực đa ngữ có sẵn |
| 5 | SigLIP 2 zero-shot + dịch máy truy vấn Việt→Anh | Baseline cạnh tranh quan trọng (bản gốc) |
| 6 | **ViSigLIP-OT / ViCLIP-OT zero-shot** (bản ngữ tiếng Việt) *— mới, 08/2026* | **Baseline cạnh tranh quan trọng nhất** — không cần dịch máy, đo trực tiếp năng lực bản ngữ |
| 7 | **SigLIP 2 + LoRA/DoRA fine-tune (mô hình đề xuất)** | Mô hình chính |
| *(tham chiếu)* | Qwen3-VL-Embedding-2B zero-shot | Mốc SOTA để biết còn cách bao xa — **không** tính vào so sánh công bằng vì kiến trúc khác biệt lớn (Mục 2.5) |

Nhóm nhấn mạnh hai hệ thống **#5 và #6**: đây là các phương án rẻ nhất (không cần huấn luyện), và nếu một trong hai cho kết quả tương đương hoặc tốt hơn mô hình #7 thì đó vẫn là **một kết quả có giá trị và nhóm sẽ báo cáo trung thực** — vì nó chứng minh rằng với miền bài toán này, mô hình có sẵn (có/không cần dịch máy) là giải pháp thực dụng hơn fine-tune. Nhóm cố ý không đặt ra chỉ tiêu kiểu "phải tăng 12–15% Recall@5", để tránh áp lực làm sai lệch kết quả thực nghiệm.

### 5.3. Ablation

| Thí nghiệm | Câu hỏi cần trả lời |
|---|---|
| **DoRA bật/tắt (cùng rank)** *— mới, 08/2026* | DoRA có thực sự cải thiện so với LoRA thường trên bài toán này, như paper gốc tuyên bố? |
| LoRA rank ∈ {4, 8, 16} | Rank cao hơn có tốt hơn, hay chỉ overfit? |
| Có / không hard negative mining | Mẫu âm khó đóng góp bao nhiêu? |
| Batch hiệu dụng ∈ {32, 128, 256} | Contrastive learning phụ thuộc batch size đến mức nào? |
| Chỉ adapter ảnh / chỉ adapter text / cả hai | Vấn đề chủ yếu nằm ở phía ngôn ngữ hay phía thị giác? |
| SigLIP 1 vs SigLIP 2 (cùng cấu hình) | Nâng cấp backbone mang lại bao nhiêu? |
| **SigLIP 2 vs ViSigLIP-OT (cùng cấu hình)** *— mới, 08/2026* | Backbone bản ngữ tiếng Việt mang lại bao nhiêu so với backbone đa ngữ tổng quát? |

*(7 trục ablation — khối lượng đầy đủ được giữ nguyên so với đề cương gốc, cộng thêm 2 trục mới từ Mục 2.5. Đây là phần rủi ro cao nhất về thời gian GPU trong lịch 5 tuần — xem thứ tự ưu tiên nếu thiếu giờ máy ở `KeHoach_Sprint_MultimodalFashionSearch.md`.)*

### 5.4. Phân tích lỗi định tính

Nhóm sẽ chọn 15–20 truy vấn có kết quả kém nhất, chụp ảnh màn hình kết quả, và phân loại nguyên nhân theo các nhóm dự kiến sau:

- Từ vựng thời trang bản địa hoặc từ mượn mà mô hình không hiểu (`croptop`, `baggy`, `ulzzang`, `váy xoè công sở`)
- Sai thuộc tính màu sắc
- Sai chất liệu (`chiffon` vs `cotton` — khó phân biệt qua ảnh)
- Ảnh chứa nhiều sản phẩm cùng lúc (bộ outfit đầy đủ)
- Ảnh có watermark, chữ, hoặc collage ghép nhiều ảnh
- Truy vấn theo phong cách trừu tượng thay vì thuộc tính cụ thể

### 5.5. Về ý nghĩa thống kê

Với khoảng 150–200 truy vấn, chênh lệch nhỏ (1–2%) giữa hai hệ thống **không** đủ để kết luận hệ thống nào tốt hơn. Nhóm sẽ chạy huấn luyện với 3 seed khác nhau nếu thời gian cho phép, báo cáo trung bình ± độ lệch chuẩn, và dùng paired bootstrap test khi so sánh hai hệ thống.

---

## 6. Thảo luận, rủi ro và đạo đức

### 6.1. Hạn chế đã lường trước

- **Cập nhật 08/2026:** dữ liệu nay lấy từ bộ công khai (chủ yếu phương Tây) thay vì một sàn TMĐT Việt Nam — kết quả càng khó tổng quát hoá sang thị trường Việt Nam thật, và **các danh mục trang phục truyền thống Việt Nam (áo dài, áo bà ba) có thể không có mẫu nào trong catalog** (xem Mục 3.2) — nhóm sẽ nêu rõ đây là hạn chế của việc đổi sang dữ liệu công khai để tiết kiệm thời gian, không phải hạn chế của phương pháp.
- Nhãn liên quan do chính nhóm chấm → có nguy cơ thiên lệch; nhóm giảm thiểu bằng chấm blind và chấm đôi, nhưng không loại bỏ hoàn toàn được.
- Chỉ hỗ trợ truy vấn đơn mô thức (text hoặc ảnh), chưa hỗ trợ truy vấn kết hợp ("ảnh này nhưng màu đỏ").
- Quy mô catalog từ dataset công khai nhỏ hơn nhiều so với catalog thực tế (hàng chục triệu), nên kết luận về độ trễ không suy rộng trực tiếp được.

### 6.2. Rủi ro triển khai

| Rủi ro | Mức độ | Phương án dự phòng |
|---|---|---|
| Dataset công khai đã chọn thiếu ảnh cho một số danh mục thời trang Việt Nam quan trọng | Cao *(mới, do đổi sang dữ liệu công khai)* | Loại các danh mục 0 mẫu khỏi Recall định lượng, chỉ báo cáo quan sát định tính; nêu rõ trong hạn chế (6.1) |
| Không đủ VRAM khi huấn luyện | Trung bình | Giảm độ phân giải ảnh, tăng gradient accumulation, giảm LoRA rank |
| Fine-tune không cải thiện được so với zero-shot / ViSigLIP-OT | Trung bình | Đây vẫn là kết quả báo cáo được; đổi trọng tâm bài sang phân tích *vì sao* không cải thiện |
| Việc chấm nhãn chiếm quá nhiều thời gian | Cao | Giảm tập truy vấn xuống 100, giảm độ sâu pool từ top-20 xuống top-10 |
| Thành viên không đồng đều về tiến độ | Trung bình | Họp tiến độ hàng tuần, mọi công việc theo dõi trên bảng kế hoạch |

### 6.3. Vấn đề đạo đức

**Thu thập dữ liệu và bản quyền.** *(Cập nhật 08/2026: không còn tự crawl — dùng dataset công khai có giấy phép nghiên cứu.)* Ảnh sản phẩm trong các dataset này vẫn thuộc quyền của thương hiệu/nguồn gốc gốc. Nhóm chỉ dùng cho mục đích học tập, tuân theo license của từng dataset, không phân phối lại ảnh ngoài phạm vi license cho phép, và sẽ nêu rõ nguồn trong báo cáo.

**Dữ liệu cá nhân.** Ảnh sản phẩm thời trang **thường có người mẫu**, tức là dữ liệu chứa hình ảnh khuôn mặt và cơ thể người thật. Với dataset công khai, vấn đề đồng ý đã do bên xuất bản dataset xử lý ở mức license, nhưng nhóm vẫn áp dụng biện pháp giảm thiểu khi dùng lại: không dùng ảnh để nhận dạng người, và **cân nhắc làm mờ khuôn mặt trong các ảnh minh hoạ đưa vào báo cáo và slide**.

**Thiên lệch của mô hình.** Mô hình vision-language tiền huấn luyện chủ yếu trên dữ liệu phương Tây. Hai loại thiên lệch nhóm sẽ chủ động đo:

- **Thiên lệch văn hoá:** mô hình có xử lý kém với trang phục truyền thống Việt Nam (áo dài, áo bà ba) so với trang phục phương Tây không? *(Cập nhật 08/2026 — xem Mục 3.2/6.1: vì catalog từ dataset công khai có thể không có mẫu áo dài/áo bà ba nào, nhóm sẽ đo thiên lệch này ở mức **zero-shot trên các baseline** (không cần catalog có đáp án đúng — quan sát mô hình trả về gì khi không có kết quả đúng) thay vì Recall tách theo danh mục như dự kiến ban đầu.)*
- **Thiên lệch hình thể:** với cùng một truy vấn, kết quả trả về có nghiêng hẳn về người mẫu gầy, da sáng không? Nhóm sẽ khảo sát định tính trên một tập truy vấn nhỏ và báo cáo quan sát, dù chưa đủ điều kiện để đo định lượng nghiêm ngặt.

**Tác động khi triển khai thật.** Nếu hệ thống này được dùng thật, nó có thể vô tình đẩy hàng giả/hàng nhái lên top (vì ảnh hàng nhái thường chính là ảnh sản phẩm thật bị lấy lại), và index cũ có thể trả về sản phẩm đã hết hàng. Nhóm nêu ra như hạn chế cần xử lý ở tầng sản phẩm, ngoài phạm vi bài tập lớn.

### 6.4. Hướng phát triển lên đồ án tốt nghiệp

1. **Truy vấn kết hợp (composed retrieval):** cho phép "ảnh này nhưng đổi màu đỏ và dài hơn". Đây là hướng thú vị nhưng nhóm ý thức rằng lĩnh vực này đã được nghiên cứu rất sâu (benchmark FashionIQ, CIRR, CIRCO; các phương pháp Pic2Word, SEARLE, FineCIR), nên nếu làm sẽ cần chọn góc tiếp cận hẹp và có đối chiếu nghiêm túc với các công trình đã có.
2. **Chưng cất text encoder tiếng Việt** để tăng chất lượng truy vấn tiếng Việt mà không tăng chi phí suy luận.
3. **Xây dựng và công bố bộ benchmark truy hồi đa mô thức tiếng Việt** — nhóm cho rằng đây là hướng khả thi nhất để có công bố khoa học, vì đóng góp nằm ở dữ liệu và phân tích hệ thống, không đòi hỏi phải vượt SOTA bằng phần cứng hạn chế.

---

## 7. Tổng hợp câu hỏi xin ý kiến giảng viên

1. **Phạm vi:** giới hạn ở truy vấn đơn mô thức (text→ảnh, ảnh→ảnh) và bỏ hoàn toàn truy vấn kết hợp ở bài tập lớn — như vậy có đủ khối lượng không?
2. ~~**Dữ liệu:** nhà trường có quy định về việc thu thập dữ liệu từ sàn TMĐT không?~~ *(Không còn cần hỏi — 08/2026 nhóm đã quyết định chỉ dùng dataset công khai, xem Mục 3.2. Câu hỏi mới thay vào: nhóm dự định dùng FashionGen/DeepFashion/Fashionpedia, thầy/cô thấy lựa chọn nào trong số này phù hợp hơn cho mục tiêu của bài, hay có gợi ý bộ khác không?)*
3. **Đánh giá:** quy trình tự soạn 150–200 truy vấn và tự chấm nhãn có được chấp nhận, hay có cách nào nhẹ hơn mà vẫn đủ chặt chẽ?
4. **Baseline:** với phát hiện mới ViSigLIP-OT (mô hình bản ngữ tiếng Việt, Mục 2.5), nhóm hiện có 6 hệ thống so sánh + 1 mô hình đề xuất — có phải là quá nhiều so với yêu cầu không, hay nên bỏ bớt hệ thống nào?
5. **Kết quả âm:** nếu mô hình fine-tune **không** tốt hơn các baseline zero-shot (dịch máy hoặc ViSigLIP-OT), việc báo cáo trung thực kết quả đó kèm phân tích nguyên nhân có được đánh giá tương đương một kết quả dương không?
6. ~~**Hướng mở rộng:** ViSigLIP-OT nên làm backbone chính hay chỉ baseline đối chiếu?~~ *(Đã tự quyết: chưa đủ căn cứ để chọn trước — SigLIP 2 vẫn là backbone chính để fine-tune chắc chắn xong, ViSigLIP-OT đo zero-shot ở Sprint 2 và fine-tune thử ở Sprint 4 nếu còn giờ máy (ablation ưu tiên #3); quyết định cuối dựa trên số liệu thật, không đoán trước.)*
7. **Hình thức:** định dạng trích dẫn tài liệu tham khảo mà thầy/cô yêu cầu là IEEE hay APA?
8. **Chi tiết trong tài liệu hướng dẫn:** nhóm thấy có chỗ ghi khổ giấy A4 và chỗ khác ghi A3 — xin thầy/cô xác nhận khổ giấy đúng của báo cáo.

