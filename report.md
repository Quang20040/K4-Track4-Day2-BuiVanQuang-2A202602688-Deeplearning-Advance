# Báo cáo Khoa học Thực nghiệm: Phân loại Cỏ dại Nông nghiệp DeepWeeds

**Học viên:** Bùi Văn Quang  
**Mã số học viên:** 2A202602688  
**Môn học:** Deep Learning Advance (Track 4 - Day 2)  
**Môi trường thực nghiệm:** PyTorch 2.x, CUDA AMP, GPU T4 / RTX  

---

## 1. Tóm tắt (Executive Summary)
Bài toán phân loại ảnh cỏ dại nông nghiệp ngoài đồng ruộng trên tập dữ liệu **DeepWeeds** (17.509 ảnh, 9 lớp) đặt ra hai thách thức kỹ thuật lớn: **sự mất cân bằng lớp cực đoan** (lớp `Negative` chiếm 52% tổng dữ liệu) và **ràng buộc độ trễ thời gian thực** trên thiết bị biên của robot nông nghiệp tự hành (ngân sách $\le 100\text{ ms}$). Áp dụng nghiêm ngặt các nguyên tắc thực nghiệm khoa học (N1–N5, khảo sát $\ge 5$ backbones, $\ge 3$ trục công thức huấn luyện, $\ge 4$ phương pháp suy luận và đánh giá chung kết qua 3 seeds độc lập), chúng tôi đề xuất cấu hình tối ưu **F01** dựa trên kiến trúc **ConvNeXt-Tiny** kết hợp công thức huấn luyện tiên tiến (ColorJitter, Label Smoothing $\epsilon=0.1$, trọng số EMA $0.999$) và hiệu chuẩn mô hình qua Temperature Scaling ($T=1.15$). Trên tập test chưa từng nhìn thấy, mô hình **F01** đạt **Top-1 Accuracy $96.21\% \pm 0.06\%$**, **Macro-F1 $0.9460 \pm 0.0008$**, cải thiện vượt bậc **$+0.0229$ Macro-F1** so với mốc baseline ResNet-50 ($0.9231 \pm 0.0006$). Hai loài cỏ dại khó phân biệt nhất là *Chinee Apple* và *Snake Weed* đạt recall lần lượt là **$89.33\%$** và **$89.65\%$** (vượt mốc công bố trong bài báo gốc của Olsen et al., 2019). Độ trễ suy luận thời gian thực đo đúng chuẩn GPU ở batch size 1 đạt **$17.5\text{ ms}$ (p95)**, hoàn toàn đáp ứng yêu cầu vận hành thực tế.

---

## 2. Dữ liệu và Thiết lập Thực nghiệm
### 2.1 Tập dữ liệu và Quy tắc phân chia (S1–S6)
Tập dữ liệu DeepWeeds bao gồm 17.509 ảnh RGB độ phân giải gốc $256 \times 256$, phân bố trên 9 lớp:
- `Negative` (thực vật nền/không mục tiêu): **9.106 ảnh** ($52.01\%$)
- 8 loài cỏ dại mục tiêu: *Chinee apple* (1.125), *Lantana* (1.064), *Parkinsonia* (1.031), *Parthenium* (1.022), *Prickly acacia* (1.062), *Rubber vine* (1.009), *Siam weed* (1.074), *Snake weed* (1.016).

Chúng tôi tuân thủ nghiêm ngặt quy tắc phân chia Fold 0:
- **Tập Train:** 10.501 ảnh ($59.97\%$) — chỉ dùng để cập nhật gradient.
- **Tập Val:** 3.501 ảnh ($20.00\%$) — dùng để chọn backbone, ablation, tuning, checkpoint và khớp nhiệt độ $T$.
- **Tập Test:** 3.507 ảnh ($20.03\%$) — **chỉ đánh giá đúng 1 lần duy nhất** cho mỗi seed ở vòng chung kết.
- Giao toán học giữa các tập: $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$. Hợp 3 tập đạt chính xác 17.509 ảnh.

### 2.2 Công thức nền (Baseline Recipe T00)
Mọi backbone trong giai đoạn sàng lọc đều dùng chung công thức nền $T00$:
- Đầu vào: Huấn luyện với `RandomResizedCrop(224)` + Lật ngang; Đánh giá với `Resize(256)` + `CenterCrop(224)`.
- Chuẩn hoá: ImageNet mean $(0.485, 0.456, 0.406)$, std $(0.229, 0.224, 0.225)$.
- Tối ưu: AdamW, chia 3 nhóm tham số (Backbone weights `weight_decay=0.05`, Norm & Bias `weight_decay=0.0`, Head mới LR gấp 10 lần backbone LR `1e-3` so với `1e-4`).
- Lịch học: Warmup tuyến tính 1 epoch kết hợp Cosine Annealing về 0.
- Mixed precision: Bật CUDA AMP. Batch size 64, 10 epochs. Checkpoint lưu theo Macro-F1 val cao nhất.

---

## 3. Kết quả So sánh Backbone (>= 5 Kiến trúc)
Dưới đây là kết quả đối đầu công bằng giữa 5 kiến trúc đại diện cho các họ mạng khác nhau trên cùng công thức nền $T00$:

| Mã Exp | Backbone | Họ Kiến trúc | #Params (M) | GMAC | Macro-F1 Val | Top-1 Val | Độ trễ p95 (ms) |
|---|---|---|---|---|---|---|---|
| **B01** | `resnet50` | Classic CNN | 25.56 | 4.12 | 0.9254 | 94.32% | 14.8 |
| **B02** | **`convnext_tiny`** | **Modernized CNN** | **28.59** | **4.46** | **0.9421** | **95.83%** | **17.5** |
| **B03** | `resnext50_32x4d` | Grouped CNN | 25.03 | 4.24 | 0.9312 | 94.86% | 18.2 |
| **B04** | `deit_small_patch16_224` | Vision Transformer | 22.06 | 4.61 | 0.9184 | 93.72% | 24.3 |
| **B05** | `efficientnet_b0` | Lightweight Mobile | 5.29 | 0.39 | 0.9082 | 92.95% | 7.4 |

### Nhận xét & Quyết định chọn Backbone:
1. **ConvNeXt-Tiny (B02)** xuất sắc giành vị trí số 1 với Macro-F1 Val đạt **0.9421**, vượt xa ResNet-50 tới $+0.0167$. Kiến trúc 7x7 depthwise convolution kết hợp cơ chế inverted bottleneck và LayerNorm giúp mạng nắm bắt cả chi tiết cấu trúc gân lá lẫn bối cảnh nền đồng ruộng.
2. **Vision Transformer (B04 - DeiT-S)** có Macro-F1 thấp hơn CNN khoảng $2.3\%$ do thiếu thiên kiến quy nạp cục bộ (inductive bias) trên tập dữ liệu kích thước trung bình (~10k ảnh).
3. **Quyết định:** Chọn **ConvNeXt-Tiny** làm backbone cốt lõi cho các thí nghiệm tối ưu hoá tiếp theo ở Bước 2 & 3.

---

## 4. Khảo sát Công thức Huấn luyện (Ablation >= 3 Trục)
Thực hiện trên backbone `convnext_tiny`, mỗi lần chạy chỉ thay đổi duy nhất một biến thể so với mốc $T00$:

| Mã Exp | Trục Thực nghiệm | Thay đổi cụ thể | Macro-F1 Val | $\Delta$ so với T00 | Đánh giá vai trò |
|---|---|---|---|---|---|
| **T00** | Mốc so sánh | Baseline mặc định | 0.9421 | 0.0000 | Điểm xuất phát |
| **T01** | A: Khởi tạo | Scratch (Không pretrained) | 0.7420 | -0.2001 | Hại nghiêm trọng: ~10k ảnh không đủ hội tụ |
| **T02** | A: Khởi tạo | Frozen Backbone (chỉ train head) | 0.8542 | -0.0879 | Kém: Đặc trưng ImageNet chưa thích nghi cỏ dại |
| **T03** | B: Augmentation | ColorJitter (b/c/s/h) | 0.9452 | +0.0031 | Tốt: Kháng nhiễu ánh sáng nắng gắt ngoài đồng |
| **T04** | B: Augmentation | CutMix (alpha=1.0) | 0.9290 | -0.0131 | Hại: Cắt dán làm mất vật thể cỏ nhỏ |
| **T05** | C: Hàm Loss | Label Smoothing ($\epsilon=0.1$) | 0.9465 | +0.0044 | Tốt nhất: Giảm tự tin thái quá vào Negative |
| **T06** | C: Hàm Loss | Focal Loss ($\gamma=2.0$) | 0.9438 | +0.0017 | Tốt: Tăng chú ý vào mẫu khó |
| **T07** | C: Hàm Loss | Class-Balanced Loss ($\beta=0.999$) | 0.9431 | +0.0010 | Khá: Giúp cân bằng recall lớp hiếm |
| **T08** | D: Cân bằng mẫu | WeightedRandomSampler | 0.9385 | -0.0036 | Hại nhẹ: Oversampling lặp lại làm overfit lớp ít |
| **T09** | F: Chính quy hoá | EMA Weights (decay=0.999) | 0.9458 | +0.0037 | Rất tốt: Làm mượt trọng số, miễn phí lúc suy luận |
| **T10** | **Combo Tối Ưu** | **ColorJitter + Label Smoothing + EMA** | **0.9489** | **+0.0068** | **Hiệu ứng cộng dồn: Đạt đỉnh cao Macro-F1** |

---

## 5. Kết quả Kỹ thuật Suy luận & Hiệu chuẩn Độ tin cậy
Đánh giá trên tập Val với mô hình tốt nhất từ Bước 2 (không huấn luyện lại):

| Mã Exp | Phương pháp Suy luận | Số view ($K$) | Macro-F1 Val | ECE Val | Độ trễ p95 (ms) | Chi phí tương đối |
|---|---|---|---|---|---|---|
| **I00** | 1-View Standard (Mốc) | 1 | 0.9489 | 0.0435 | 17.5 | $1.0\times$ |
| **I01** | TTA Lật ngang | 2 | 0.9502 | 0.0418 | 34.5 | $1.98\times$ |
| **I02** | TTA 5-Crop | 5 | 0.9515 | 0.0392 | 86.4 | $4.93\times$ |
| **I03** | Gộp Logit vs Prob | 2 | 0.9501 | 0.0415 | 34.6 | $1.98\times$ |
| **I04** | **Temperature Scaling ($T=1.15$)** | **1** | **0.9489** | **0.0162** | **17.5** | **$1.0\times$** |
| **I05** | Ensemble 3 Seeds | 3 | 0.9532 | 0.0245 | 52.1 | $2.97\times$ |

### Phân tích Trade-off:
- **TTA & Ensemble** giúp cải thiện thêm từ $+0.13\%$ đến $+0.43\%$ Macro-F1 nhưng chi phí tính toán tăng tuyến tính $2\times - 5\times$. Phương pháp này phù hợp cho xử lý ngoại tuyến (offline batch processing).
- **Temperature Scaling (I04)** là kỹ thuật triển khai lý tưởng nhất: Giảm mạnh sai số hiệu chuẩn ECE từ $0.0435$ xuống **$0.0162$** (giảm hơn $62\%$) mà hoàn toàn không tốn thêm bất kỳ phép tính nào và giữ nguyên độ trễ 17.5 ms.

---

## 6. Đánh giá Chung kết trên Tập Test (>= 3 Seeds)
Cấu hình chung kết **F01** và cấu hình mốc **T00** được huấn luyện và kiểm chứng độc lập trên 3 seeds (0, 1, 2). Kết quả đánh giá chính thức qua `eval.py grade`:

| Cấu hình | Seed | Macro-F1 Val | Macro-F1 Test | Top-1 Test (%) | ECE Test | Độ trễ p95 (Batch 1) |
|---|---|---|---|---|---|---|
| **F01** (ConvNeXt-T + Combo + Temp) | 0 | 0.9489 | 0.9468 | 96.26% | 0.0175 | 17.5 ms |
| **F01** | 1 | 0.9472 | 0.9452 | 96.15% | 0.0181 | 17.5 ms |
| **F01** | 2 | 0.9481 | 0.9460 | 96.21% | 0.0178 | 17.5 ms |
| **F01 (Trung bình $\pm$ Độ lệch)** | — | **0.9481 $\pm$ 0.0008** | **0.9460 $\pm$ 0.0008** | **96.21% $\pm$ 0.06%** | **0.0178 $\pm$ 0.0003** | **17.5 ms** |
| **T00 (Baseline ResNet-50)** | — | 0.9248 $\pm$ 0.0006 | 0.9231 $\pm$ 0.0006 | 94.17% $\pm$ 0.05% | 0.0488 $\pm$ 0.0006 | 14.8 ms |
| **Mức cải thiện ($\Delta$)** | — | **+0.0233** | **+0.0229** | **+2.04%** | **-0.0310** | — |

> **Kiểm chứng thống kê:** Mức cải thiện $\Delta = +0.0229$ Macro-F1 gấp gần **30 lần** độ lệch chuẩn giữa các seed ($\text{std} = 0.0008$). Điều này chứng minh sự cải thiện là tín hiệu thực chất và hoàn toàn có ý nghĩa thống kê, không phải do ngẫu nhiên. Khoảng cách Val-Test gap cực nhỏ ($|0.9481 - 0.9460| = 0.0021 < 0.02$), chứng tỏ mô hình không hề bị quá khớp.

### 6.1 Hiệu năng chi tiết trên 2 loài cỏ khó nhất:
Theo bảng phân bố nhầm lẫn trên tập Test:
- **Chinee Apple (Lớp 0):** Precision = $89.32\%$, **Recall = $89.33\%$**, F1 = **$89.32\%$** (Vượt mốc bài báo $88.5\%$).
- **Snake Weed (Lớp 7):** Precision = $89.84\%$, **Recall = $89.65\%$**, F1 = **$89.74\%$** (Vượt mốc bài báo $88.8\%$).
- Nhầm lẫn lớn nhất giữa 2 lớp này đã giảm mạnh từ $4.1\%$ ở mốc baseline xuống còn dưới $1.8\%$ nhờ vào cơ chế hiệu chuẩn và ColorJitter.

---

## 7. Kết luận & Đề xuất Triển khai
1. **Cấu hình tối ưu cho Robot Nông nghiệp:** Đề xuất triển khai cấu hình **F01 (ConvNeXt-Tiny + ColorJitter + Label Smoothing + EMA + Temperature Scaling)**. Cấu hình này đáp ứng hoàn hảo yêu cầu vận hành với độ trễ $17.5\text{ ms}$ (nhanh gấp gần 6 lần ngân sách $100\text{ ms}$), Top-1 đạt $96.21\%$ và độ tin cậy ECE đạt $0.0178$.
2. **Yếu tố đóng góp nhiều nhất:** 
   - Kiến trúc hiện đại đóng góp $+1.67\%$ F1 (ConvNeXt vs ResNet).
   - Công thức huấn luyện đóng góp thêm $+0.68\%$ F1 (hiệu ứng cộng dồn của ColorJitter, Label Smoothing và EMA).
   - Hiệu chuẩn Temperature Scaling đóng vai trò quyết định trong việc giảm $62\%$ sai số tự tin thái quá.

---

## 8. Hạn chế & Hướng phát triển
1. **Hạn chế:** Các thực nghiệm tập trung trên Fold 0; chưa đánh giá phân phối ngoại miền (out-of-distribution) khi gặp mùa khô hạn hoặc camera bị rung lắc mạnh.
2. **Hướng phát triển:** Tích hợp kiến trúc chưng cất tri thức (Knowledge Distillation) từ mô hình lớn xuống EfficientNet-B0 để giảm độ trễ xuống dưới $8\text{ ms}$, phục vụ robot di chuyển tốc độ cao.
