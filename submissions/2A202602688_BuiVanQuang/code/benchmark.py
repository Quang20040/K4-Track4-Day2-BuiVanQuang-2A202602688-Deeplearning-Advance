"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    - chạy warmup lần đầu rồi bỏ
    - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
    - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo đạc
    times_ms = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.array(times_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(times_arr, 50)),
        "p95": float(np.percentile(times_arr, 95)),
        "p99": float(np.percentile(times_arr, 99)),
        "mean": float(np.mean(times_arr)),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}
    """
    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    sync = torch.cuda.synchronize if use_cuda else None
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"

    model_eval = model.to(dev)
    model_eval.eval()

    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16":
        model_eval = model_eval.half()
        x = x.half()

    def _forward_fn():
        with torch.inference_mode():
            if dtype == "amp" and use_cuda:
                with torch.autocast(device_type="cuda"):
                    model_eval(x)
            else:
                model_eval(x)

    res = bench(_forward_fn, warmup=warmup, iters=iters, sync=sync)
    p50_ms = res["p50"]
    images_per_s = float(batch_size / (p50_ms / 1000.0)) if p50_ms > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": p50_ms,
        "p95": res["p95"],
        "p99": res["p99"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ thực tế của TTA K views và so sánh với K * p50 của 1 view."""
    # Đo 1 view
    single_res = latency_report(
        model, batch_size=batch_size, img_size=img_size, dtype=dtype, device=device,
        warmup=warmup, iters=iters,
    )

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    sync = torch.cuda.synchronize if use_cuda else None

    model_eval = model.to(dev)
    model_eval.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        model_eval = model_eval.half()
        x = x.half()

    def _k_forward_fn():
        with torch.inference_mode():
            for _ in range(k_views):
                if dtype == "amp" and use_cuda:
                    with torch.autocast(device_type="cuda"):
                        model_eval(x)
                else:
                    model_eval(x)

    multi_res = bench(_k_forward_fn, warmup=warmup, iters=iters, sync=sync)

    return {
        "k_views": k_views,
        "single_p50": single_res["p50"],
        "tta_p50": multi_res["p50"],
        "tta_p95": multi_res["p95"],
        "tta_p99": multi_res["p99"],
        "k_times_single": k_views * single_res["p50"],
        "overhead_ratio": multi_res["p50"] / max(single_res["p50"], 1e-4),
    }

