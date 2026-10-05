"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model: nn.Module, loader, device: torch.device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    """
    model.eval()
    all_fnames = []
    all_y = []
    all_logits = []

    with torch.inference_mode():
        for images, targets, fnames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)

            logits = model(images)

            all_fnames.extend(fnames)
            all_y.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_y, axis=0) if all_y else np.array([])
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    return all_fnames, y_true, logits


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> list[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop`."""
    _, _, h, w = x.shape
    top_left = x[:, :, 0:crop, 0:crop]
    top_right = x[:, :, 0:crop, w - crop:w]
    bottom_left = x[:, :, h - crop:h, 0:crop]
    bottom_right = x[:, :, h - crop:h, w - crop:w]
    ch = (h - crop) // 2
    cw = (w - crop) // 2
    center = x[:, :, ch:ch + crop, cw:cw + crop]
    return [top_left, top_right, bottom_left, bottom_right, center]


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    views = []
    for s in sizes:
        resized = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        views.append(resized)
    return views


def _softmax(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def aggregate_views(logits_per_view: list[np.ndarray | torch.Tensor], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    numpy_logits = [
        l.cpu().numpy() if isinstance(l, torch.Tensor) else np.asarray(l, dtype=np.float64)
        for l in logits_per_view
    ]

    if space == "prob":
        probs_per_view = [_softmax(l) for l in numpy_logits]
        mean_probs = np.mean(probs_per_view, axis=0)
        return mean_probs / np.sum(mean_probs, axis=-1, keepdims=True)
    elif space == "logit":
        mean_logits = np.mean(numpy_logits, axis=0)
        return _softmax(mean_logits)
    else:
        raise ValueError(f"Không hỗ trợ space: {space} (chỉ 'prob' hoặc 'logit')")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    mean_p = np.mean(list_of_probs, axis=0)
    return mean_p / np.sum(mean_p, axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray | torch.Tensor, val_labels: np.ndarray | torch.Tensor) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69)."""
    logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    labels_t = torch.as_tensor(val_labels, dtype=torch.long)

    # Tối ưu hoá log(T) để T luôn dương
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t)
        loss = criterion(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    best_t = float(torch.exp(log_t).item())
    return max(best_t, 1e-3)


def apply_temperature(logits: np.ndarray | torch.Tensor, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    z = logits.cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits, dtype=np.float64)
    scaled = z / max(float(T), 1e-4)
    return _softmax(scaled)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75)."""
    fused_model = copy.deepcopy(model)
    fused_model.eval()

    # Sử dụng tiện ích gộp conv-bn có sẵn của PyTorch
    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval
    except ImportError:
        fuse_conv_bn_eval = None

    def _fuse_children(parent: nn.Module):
        prev_name = None
        prev_child = None

        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)) and isinstance(prev_child, nn.Conv2d):
                if fuse_conv_bn_eval is not None:
                    fused_conv = fuse_conv_bn_eval(prev_child, child)
                    setattr(parent, prev_name, fused_conv)
                    setattr(parent, name, nn.Identity())
                    prev_name = None
                    prev_child = None
                    continue
            _fuse_children(child)
            prev_name = name
            prev_child = child

    _fuse_children(fused_model)
    return fused_model

