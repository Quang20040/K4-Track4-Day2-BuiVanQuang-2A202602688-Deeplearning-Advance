import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Đọc các file code đã hoàn thiện để nhúng trực tiếp hoặc import
dataset_py = (ROOT / "starter" / "dataset.py").read_text(encoding="utf-8")
model_py = (ROOT / "starter" / "model.py").read_text(encoding="utf-8")
losses_py = (ROOT / "starter" / "losses.py").read_text(encoding="utf-8")
train_py = (ROOT / "starter" / "train.py").read_text(encoding="utf-8")
inference_py = (ROOT / "starter" / "inference.py").read_text(encoding="utf-8")
benchmark_py = (ROOT / "starter" / "benchmark.py").read_text(encoding="utf-8")
eval_py = (ROOT / "eval.py").read_text(encoding="utf-8")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Lab Day 2 — Deep Learning Advance: Backbone, Training Recipes & Inference\n",
            "**Dự án:** Phân loại Cỏ dại DeepWeeds cho Robot Nông nghiệp Tự hành\n",
            "**Sinh viên:** Bùi Văn Quang — **MSSV:** 2A202602688\n",
            "\n",
            "Notebook này được đóng gói hoàn chỉnh trọn gói (Self-contained) để chạy mượt mà trên **Google Colab (GPU T4)**.\n",
            "Chỉ cần bấm **Runtime -> Run all** để tự động thực thi từ Bước 0 đến Bước 5!"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Cài đặt Môi trường & Kiểm tra GPU"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "!pip install -q timm openpyxl matplotlib pandas scikit-learn\n",
            "\n",
            "import os, sys, platform, torch\n",
            "print(\"Python:\", platform.python_version())\n",
            "print(\"PyTorch:\", torch.__version__)\n",
            "device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (CẢNH BÁO: Hãy vào Runtime -> Change runtime type -> Chọn T4 GPU)'\n",
            "print(\"Phần cứng:\", device_name)\n",
            "assert torch.cuda.is_available(), 'Vui lòng bật GPU trong Runtime -> Change runtime type -> T4 GPU trước khi chạy!'"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Tải Dữ liệu DeepWeeds (Ảnh & Nhãn Fold 0)"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import urllib.request, zipfile\n",
            "from pathlib import Path\n",
            "\n",
            "os.makedirs(\"data/images\", exist_ok=True)\n",
            "os.makedirs(\"data/labels\", exist_ok=True)\n",
            "\n",
            "# Tải 4 file nhãn CSV từ GitHub DeepWeeds chính thức\n",
            "base_url = \"https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels\"\n",
            "csv_files = [\"labels.csv\", \"train_subset0.csv\", \"val_subset0.csv\", \"test_subset0.csv\"]\n",
            "for fname in csv_files:\n",
            "    dest = f\"data/labels/{fname}\"\n",
            "    if not os.path.exists(dest):\n",
            "        print(f\"Đang tải {fname}...\")\n",
            "        urllib.request.urlretrieve(f\"{base_url}/{fname}\", dest)\n",
            "\n",
            "# Tải images.zip từ Zenodo và giải nén trực tiếp vào ổ SSD Colab\n",
            "if not os.path.exists(\"data/images.zip\"):\n",
            "    print(\"Đang tải images.zip (~468.7 MB) từ Zenodo...\")\n",
            "    !wget -q -O data/images.zip \"https://zenodo.org/records/7939060/files/images.zip?download=1\"\n",
            "\n",
            "print(\"Đang giải nén ảnh...\")\n",
            "!unzip -q -n data/images.zip -d data/images/\n",
            "\n",
            "n_imgs = len(list(Path(\"data/images\").glob(\"*.jpg\")))\n",
            "print(f\"=> ĐÃ SẴN SÀNG: {n_imgs} ảnh trong data/images/ và đủ 4 file nhãn CSV!\")\n",
            "assert n_imgs >= 17509, f\"Số ảnh chưa đủ: {n_imgs}/17509\""
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Khởi tạo Toàn bộ Mã nguồn Module Dự án"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "os.makedirs(\"starter\", exist_ok=True)\n",
            "\n",
            "# Ghi file eval.py\n",
            "with open(\"eval.py\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write('''" + eval_py.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
            "\n",
            "# Ghi starter/dataset.py\n",
            "with open(\"starter/dataset.py\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write('''" + dataset_py.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
            "\n",
            "# Ghi starter/model.py\n",
            "with open(\"starter/model.py\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write('''" + model_py.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
            "\n",
            "# Ghi starter/losses.py\n",
            "with open(\"starter/losses.py\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write('''" + losses_py.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
            "\n",
            "# Ghi starter/train.py\n",
            "with open(\"starter/train.py\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write('''" + train_py.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
            "\n",
            "# Ghi starter/inference.py\n",
            "with open(\"starter/inference.py\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write('''" + inference_py.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
            "\n",
            "# Ghi starter/benchmark.py\n",
            "with open(\"starter/benchmark.py\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write('''" + benchmark_py.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
            "\n",
            "sys.path.insert(0, \".\")\n",
            "sys.path.insert(0, \"starter\")\n",
            "print(\"=> Đã tải xong toàn bộ các modules mã nguồn vào môi trường Colab!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Tự Động Chạy Toàn Bộ Thí Nghiệm (Bước 0 đến Bước 5)"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Chạy toàn bộ pipeline tự động từ run_all_experiments.py\n",
            "run_all_code = '''" + (ROOT / "run_all_experiments.py").read_text(encoding="utf-8").replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "'''\n",
            "with open(\"run_all_experiments.py\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write(run_all_code)\n",
            "\n",
            "!python run_all_experiments.py"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Chấm Điểm Chính Thức với `eval.py`"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "print(\"=== 1. CHẤM SCORE CHO CẤU HÌNH CHUNG KẾT F01 ===\")\n",
            "!python eval.py score --pred \"predictions/F01_seed*_test.csv\" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out\n",
            "\n",
            "print(\"\\n=== 2. TỰ ĐỘNG CHẤM GRADE RUBRIC PHẦN I ===\")\n",
            "!python eval.py grade --final \"predictions/F01_seed*_test.csv\" --baseline \"predictions/T00_seed*_test.csv\" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --out eval_out"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Đóng Gói Toàn Bộ Bài Nộp (Tải về máy tính)"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Đóng gói theo cấu trúc thư mục quy định của đề bài\n",
            "SUBMISSION_DIR = \"submissions/2A202602688_BuiVanQuang\"\n",
            "os.makedirs(f\"{SUBMISSION_DIR}/curves\", exist_ok=True)\n",
            "os.makedirs(f\"{SUBMISSION_DIR}/predictions\", exist_ok=True)\n",
            "os.makedirs(f\"{SUBMISSION_DIR}/code\", exist_ok=True)\n",
            "\n",
            "!cp -rf curves/* {SUBMISSION_DIR}/curves/ 2>/dev/null || true\n",
            "!cp -rf predictions/* {SUBMISSION_DIR}/predictions/ 2>/dev/null || true\n",
            "!cp -rf starter/*.py {SUBMISSION_DIR}/code/\n",
            "!cp -f eval.py {SUBMISSION_DIR}/code/\n",
            "!cp -f results.xlsx {SUBMISSION_DIR}/\n",
            "!cp -f report.md {SUBMISSION_DIR}/\n",
            "\n",
            "# Nén thành submission.zip để người dùng tải về ngay\n",
            "!zip -r submission_2A202602688_BuiVanQuang.zip submissions/ results.xlsx report.md curves/ predictions/\n",
            "\n",
            "print(\"\\n=> HOÀN TẤT 100%! Bấm vào tab Files (biểu tượng thư mục bên trái Colab) để tải file 'submission_2A202602688_BuiVanQuang.zip' về nộp bài!\")\n",
            "\n",
            "# Tự động mở cửa sổ tải về trên trình duyệt\n",
            "from google.colab import files\n",
            "try:\n",
            "    files.download('submission_2A202602688_BuiVanQuang.zip')\n",
            "except Exception as e:\n",
            "    print('Nếu trình duyệt không tự tải, bạn hãy tải thủ công từ cây thư mục bên trái:', e)"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

target_path = ROOT / "starter" / "lab_day2.ipynb"
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Đã tạo thành công notebook tự động trọn gói: {target_path}")
