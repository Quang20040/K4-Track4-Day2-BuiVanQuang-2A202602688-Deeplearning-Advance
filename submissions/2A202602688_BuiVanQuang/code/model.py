"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = (init != "scratch") and pretrained
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Ghi nhận tag trọng số thực tế được tải
    tag = "scratch"
    if is_pretrained:
        if hasattr(model, "pretrained_cfg") and isinstance(model.pretrained_cfg, dict):
            tag = model.pretrained_cfg.get("tag", "default") or "default"
        else:
            tag = "pretrained"
    model.tag = tag
    model.frozen_backbone = False

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head.

    - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
    - backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters()) if classifier is not None else set()

    for param in model.parameters():
        if param not in head_params:
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.frozen_backbone = True
    # Chuyển các lớp chuẩn hoá sang eval
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters()) if classifier is not None else set()

    backbone_decay = []
    backbone_no_decay = []
    head_group = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in head_params:
            head_group.append(param)
        else:
            if param.ndim <= 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_group:
        groups.append({"params": head_group, "lr": lr_head, "weight_decay": weight_decay})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x)."""
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x).total()
        return float(flops) / 1e9
    except Exception:
        try:
            import thop
            macs, _ = thop.profile(model, inputs=(x,), verbose=False)
            return float(macs) / 1e9
        except Exception:
            # Fallback tính toán GMAC qua hook chuẩn
            macs = 0
            hooks = []

            def conv_hook(module, inp, out):
                nonlocal macs
                batch, c_out, h_out, w_out = out.shape
                c_in = module.in_channels // module.groups
                k_h, k_w = module.kernel_size
                macs += batch * c_out * h_out * w_out * (c_in * k_h * k_w)

            def linear_hook(module, inp, out):
                nonlocal macs
                macs += inp[0].shape[0] * module.in_features * module.out_features

            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    hooks.append(m.register_forward_hook(conv_hook))
                elif isinstance(m, nn.Linear):
                    hooks.append(m.register_forward_hook(linear_hook))

            was_training = model.training
            model.eval()
            with torch.inference_mode():
                model(x)
            if was_training:
                model.train()
            for h in hooks:
                h.remove()
            return float(macs) / 1e9

