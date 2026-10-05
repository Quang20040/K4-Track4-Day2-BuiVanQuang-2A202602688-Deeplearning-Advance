"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    num_workers: int = 0 if os.name == "nt" else 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    import model as md
    groups = md.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    import math
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup = {}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.clone().detach()
                p.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def copy_to(self, model: torch.nn.Module) -> None:
        self.apply_shadow(model)


def train_one_epoch(model: torch.nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}."""
    import losses as ls

    model.train()
    if getattr(model, "frozen_backbone", False):
        classifier = model.get_classifier()
        head_modules = set(classifier.modules()) if classifier is not None else set()
        for m in model.modules():
            if m not in head_modules:
                m.eval()

    total_loss = 0.0
    count = 0
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    use_amp = cfg.amp and (device_type == "cuda")

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        is_mixed = False
        if cfg.mix:
            images, mixed_targets = ls.mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)
            is_mixed = True

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(images)
            if is_mixed:
                loss = ls.mixed_loss(criterion, outputs, mixed_targets)
            else:
                loss = criterion(outputs, targets)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * len(targets)
        count += len(targets)

    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": total_loss / max(1, count), "lr": current_lr}


def evaluate(model: torch.nn.Module, loader, criterion, device: torch.device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    """
    model.eval()
    total_loss = 0.0
    count = 0
    all_fnames = []
    all_y = []
    all_logits = []

    with torch.inference_mode():
        for images, targets, fnames in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            count += len(targets)

            all_fnames.extend(fnames)
            all_y.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_y, axis=0) if all_y else np.array([])
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    avg_loss = total_loss / max(1, count)
    return all_fnames, y_true, logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", np.nan) for h in history]
    val_loss = [h.get("val_loss", np.nan) for h in history]
    val_f1 = [h.get("val_macro_f1", np.nan) for h in history]
    lrs = [h.get("lr", np.nan) for h in history]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    axs[0].plot(epochs, train_loss, "o-", label="Train Loss", color="royalblue")
    axs[0].plot(epochs, val_loss, "s-", label="Val Loss", color="crimson")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title(f"{title} - Loss")
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].legend()

    axs[1].plot(epochs, val_f1, "^-", label="Val Macro-F1", color="forestgreen")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Macro-F1")
    axs[1].set_title(f"{title} - Macro-F1")
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].legend()

    axs[2].plot(epochs, lrs, "d-", label="LR", color="darkorange")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Learning Rate")
    axs[2].set_title(f"{title} - Learning Rate")
    axs[2].grid(True, linestyle="--", alpha=0.6)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    import copy
    import dataclasses
    import json
    import time
    from eval import compute_metrics, save_predictions
    import dataset as ds
    import model as md
    import losses as ls

    # 1. Khởi tạo
    set_seed(cfg.seed)
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path("curves").mkdir(parents=True, exist_ok=True)

    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # 2. Dữ liệu & kiểm tra split S1-S6
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, fold=cfg.fold)
    ds.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tfm = ds.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    eval_tfm = ds.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = ds.make_loader(
        train_df, cfg.images_dir, train_tfm,
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
    )
    val_loader = ds.make_loader(
        val_df, cfg.images_dir, eval_tfm,
        batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers,
    )

    # 3. Model, Loss, Optimizer, Scheduler, EMA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = md.build_model(
        cfg.backbone, pretrained=True, num_classes=ds.NUM_CLASSES,
        drop_rate=cfg.drop_rate, init=cfg.init,
    ).to(device)

    # GMACs và Params
    n_params = md.count_params(net)
    gmacs = md.count_gmacs(net, img_size=cfg.img_size)

    # Criterion
    weight = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = train_df["Label"].value_counts().to_dict()
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        weight = ls.class_weights(counts, beta=beta).to(device)

    criterion = ls.build_criterion(
        kind=cfg.loss,
        label_smoothing=cfg.label_smoothing,
        focal_gamma=cfg.focal_gamma,
        weight=weight,
    )
    eval_criterion = torch.nn.CrossEntropyLoss()

    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and torch.cuda.is_available())
    ema = EMA(net, decay=cfg.ema_decay) if cfg.ema_decay is not None else None
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and torch.cuda.is_available())
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and torch.cuda.is_available())

    # 4. Huấn luyện qua các epoch
    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_val_logits = None
    best_val_fnames = None
    best_val_y = None
    epoch_times = []

    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN [{cfg.exp_id}] (Backbone={cfg.backbone}, Init={cfg.init}, Seed={cfg.seed}) ---")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            net, train_loader, criterion, optimizer, scheduler, scaler,
            cfg=cfg, device=device, ema=ema,
        )
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        # Đánh giá trên VAL
        if ema is not None:
            ema.apply_shadow(net)
        fnames, y_val, val_logits, v_loss = evaluate(net, val_loader, eval_criterion, device)
        if ema is not None:
            ema.restore(net)

        val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
        val_pred = val_probs.argmax(axis=1)
        val_metrics = compute_metrics(y_val, val_pred, val_probs)

        row = {
            "epoch": epoch,
            "train_loss": float(train_res["train_loss"]),
            "val_loss": float(v_loss),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_top1": float(val_metrics["top1"]),
            "val_balanced_acc": float(val_metrics["balanced_acc"]),
            "val_ece": float(val_metrics["ece"]),
            "lr": float(train_res["lr"]),
            "time_s": float(t_epoch),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
            f"Train Loss: {row['train_loss']:.4f} | "
            f"Val Loss: {row['val_loss']:.4f} | "
            f"Val Macro-F1: {row['val_macro_f1']:.4f} | "
            f"Val Top-1: {row['val_top1']*100:.2f}% | "
            f"LR: {row['lr']:.2e} | "
            f"Time: {t_epoch:.1f}s"
        )

        # Lưu checkpoint theo macro-F1 cao nhất (hòa lấy epoch sớm hơn)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            if ema is not None:
                ema.apply_shadow(net)
            best_state = copy.deepcopy(net.state_dict())
            if ema is not None:
                ema.restore(net)
            best_val_logits = val_logits
            best_val_fnames = fnames
            best_val_y = y_val

    # 5. Lưu checkpoint và dự đoán tốt nhất trên VAL
    if best_state is not None:
        net.load_state_dict(best_state)
        torch.save(best_state, r_dir / "best_model.pt")

    val_probs = torch.softmax(torch.tensor(best_val_logits), dim=-1).numpy()
    save_predictions(pred_path(cfg, "val"), best_val_fnames, best_val_y, val_probs)
    np.save(r_dir / "val_logits.npy", best_val_logits)

    # 6. Bước 4: Đánh giá TEST nếu được yêu cầu
    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = ds.make_loader(
            test_df, cfg.images_dir, eval_tfm,
            batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers,
        )
        t_fnames, y_test, test_logits, _ = evaluate(net, test_loader, eval_criterion, device)
        test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
        save_predictions(pred_path(cfg, "test"), t_fnames, y_test, test_probs)
        np.save(r_dir / "test_logits.npy", test_logits)
        test_metrics = compute_metrics(y_test, test_probs.argmax(axis=1), test_probs)
        print(f"--> KẾT QUẢ TEST: Top-1={test_metrics['top1']*100:.2f}% | Macro-F1={test_metrics['macro_f1']:.4f}")

    # 7. Xuất log history và biểu đồ
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(r_dir / "history.csv", index=False)
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png", title=f"{cfg.exp_id} ({cfg.backbone})")

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "params_m": n_params,
        "gmacs": gmacs,
        "mean_time_epoch": float(np.mean(epoch_times)),
    }
    if test_metrics is not None:
        summary["test_macro_f1"] = test_metrics["macro_f1"]
        summary["test_top1"] = test_metrics["top1"]
        summary["test_ece"] = test_metrics["ece"]

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(Config)}
    overrides = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Tham số không đúng định dạng key=val: {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in fields:
            raise KeyError(f"Trường cấu hình '{k}' không có trong Config")
        f = fields[k]
        if v.lower() in ("none", "null"):
            overrides[k] = None
        elif f.name in ("seed", "fold", "epochs", "batch_size", "img_size", "num_workers"):
            overrides[k] = int(v)
        elif f.name in ("lr_backbone", "lr_head", "weight_decay", "warmup_epochs", "mix_alpha",
                        "label_smoothing", "focal_gamma", "drop_rate", "ema_decay", "class_weight_beta"):
            overrides[k] = float(v)
        elif f.name in ("amp", "save_test_predictions"):
            overrides[k] = v.lower() in ("true", "1", "yes")
        else:
            overrides[k] = v
    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    import argparse
    import dataclasses
    parser = argparse.ArgumentParser(description="Chạy huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Cặp KEY=VALUE để ghi đè Config")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"Cấu hình chạy: {dataclasses.asdict(cfg)}")
    res = run(cfg)
    print("Kết quả:", res)


if __name__ == "__main__":
    main()

