# Bài Nộp Lab Day 2 — Deep Learning Advance

- **Sinh viên:** Bùi Văn Quang
- **MSSV:** 2A202602688
- **Dataset:** DeepWeeds (Fold 0)
- **Cấu hình tốt nhất:** F01 (ConvNeXt-Tiny + ColorJitter + Label Smoothing + EMA + Temperature Scaling)

## Cấu trúc thư mục:
- `results.xlsx`: Đầy đủ 7 sheets chuẩn hoá (Backbones, Training, Inference, Final, PerClass, Latency, Summary).
- `report.md`: Báo cáo khoa học thực nghiệm chi tiết.
- `curves/`: Chứa toàn bộ 19 biểu đồ huấn luyện của từng thử nghiệm (B01-B05, T00-T10, F01 seeds).
- `predictions/`: Chứa toàn bộ các file dự đoán CSV của vòng chung kết F01 và baseline T00 qua 3 seeds.
- `code/`: Toàn bộ mã nguồn thực thi đã hoàn thiện 100%.

## Lệnh tự chấm điểm với eval.py:
```bash
python code/eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv ../../data/labels/test_subset0.csv --labels ../../data/labels/labels.csv
python code/eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --test-csv ../../data/labels/test_subset0.csv --val-csv ../../data/labels/val_subset0.csv --labels ../../data/labels/labels.csv --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --latency-p95-ms 17.5 --latency-method proper
```
