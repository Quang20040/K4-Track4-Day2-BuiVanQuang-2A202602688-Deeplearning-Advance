"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file split: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    """
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    # 1. Giao rỗng
    train_val = train_files & val_files
    train_test = train_files & test_files
    val_test = val_files & test_files

    assert len(train_val) == 0, f"Vi phạm S4: Giao train và val không rỗng ({len(train_val)} ảnh trùng)"
    assert len(train_test) == 0, f"Vi phạm S4: Giao train và test không rỗng ({len(train_test)} ảnh trùng)"
    assert len(val_test) == 0, f"Vi phạm S4: Giao val và test không rỗng ({len(val_test)} ảnh trùng)"

    # 2. Hợp ba tập
    all_files = train_files | val_files | test_files
    assert len(all_files) == 17509, f"Vi phạm S4: Tổng số ảnh = {len(all_files)}, kỳ vọng đúng 17.509 ảnh"
    assert len(train_df) + len(val_df) + len(test_df) == 17509, "Tổng số dòng không khớp 17.509"

    # 3. Kiểm tra file tồn tại trên đĩa
    images_dir = Path(images_dir)
    missing_files = []
    if images_dir.exists():
        missing_files = [f for f in all_files if not (images_dir / f).exists()]
        assert len(missing_files) == 0, (
            f"Vi phạm S4: Có {len(missing_files)} file không tìm thấy trong {images_dir}. "
            f"Mẫu: {missing_files[:5]}"
        )

    # 4. Thống kê theo lớp
    train_counts = train_df["Label"].value_counts().to_dict()
    val_counts = val_df["Label"].value_counts().to_dict()
    test_counts = test_df["Label"].value_counts().to_dict()

    stats = {
        "n": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(all_files),
        },
        "per_class": {
            "train": {int(k): int(v) for k, v in train_counts.items()},
            "val": {int(k): int(v) for k, v in val_counts.items()},
            "test": {int(k): int(v) for k, v in test_counts.items()},
        },
        "overlap": {
            "train_val": len(train_val),
            "train_test": len(train_test),
            "val_test": len(val_test),
        },
        "missing_files_count": len(missing_files),
    }

    print("=" * 65)
    print("XÁC MINH PHÂN CHIA DỮ LIỆU (S1-S4):")
    print(f"  - Train: {len(train_df)} ({len(train_df)/17509*100:.2f}%)")
    print(f"  - Val:   {len(val_df)} ({len(val_df)/17509*100:.2f}%)")
    print(f"  - Test:  {len(test_df)} ({len(test_df)/17509*100:.2f}%)")
    print(f"  - Tổng:  {len(all_files)} ảnh (kỳ vọng: 17509) -> ĐẠT")
    print(f"  - Giao tập: train∩val={len(train_val)}, train∩test={len(train_test)}, val∩test={len(val_test)} -> ĐẠT")
    if images_dir.exists():
        print(f"  - Đĩa ảnh ({images_dir}): Toàn bộ 17509 file đều tồn tại -> ĐẠT")
    print("=" * 65)

    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.
    """
    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if aug == "basic":
        tfms = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # Cỏ dại chụp từ trên xuống (nadir), bất biến với hướng quay/lật
        ]
    elif aug == "color":
        tfms = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
    elif aug == "randaug":
        tfms = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
        ]
    elif aug == "trivial":
        tfms = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.TrivialAugmentWide(),
        ]
    else:
        tfms = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(tfms)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        fname = self.filenames[i]
        label = self.labels[i]
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label), str(fname)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader.

    - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
    - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số 1/(số ảnh của lớp)
    - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
    - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    dataset = DeepWeedsDataset(df=df, images_dir=images_dir, transform=transform)

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if train:
        if sampler == "balanced":
            counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / counts[y] for y in df["Label"]]
            weighted_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=weighted_sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=len(dataset) > batch_size,
                worker_init_fn=_seed_worker,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=len(dataset) > batch_size,
                worker_init_fn=_seed_worker,
            )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

