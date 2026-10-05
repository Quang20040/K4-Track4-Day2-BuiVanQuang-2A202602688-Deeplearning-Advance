import os
import zipfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
ZIP_PATH = ROOT / "images.zip"

def setup_data():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tải labels CSV từ GitHub DeepWeeds chính thức
    base_url = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
    csv_files = ["labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv"]

    print("--- 1. Đang tải các file metadata CSV ---")
    for fname in csv_files:
        dest = LABELS_DIR / fname
        if not dest.exists() or dest.stat().st_size == 0:
            url = f"{base_url}/{fname}"
            print(f"Đang tải {fname} từ {url}...")
            urllib.request.urlretrieve(url, dest)
            print(f"  -> Lưu vào: {dest} ({dest.stat().st_size} bytes)")
        else:
            print(f"  -> {fname} đã tồn tại ({dest.stat().st_size} bytes).")

    # 2. Giải nén images.zip
    print("\n--- 2. Đang kiểm tra và giải nén images.zip ---")
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file {ZIP_PATH}")

    print(f"Tìm thấy {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB)")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        members = zf.namelist()
        print(f"Tổng số phần tử trong zip: {len(members)}")
        
        # Kiểm tra xem đã giải nén chưa
        existing_jpgs = list(IMAGES_DIR.glob("*.jpg"))
        if len(existing_jpgs) >= 17509:
            print(f"Đã có đủ {len(existing_jpgs)} ảnh trong {IMAGES_DIR}. Bỏ qua giải nén.")
        else:
            print(f"Đang giải nén 17509 ảnh vào {IMAGES_DIR} (quá trình này có thể mất 30-60 giây)...")
            # Trích xuất trực tiếp các file ảnh
            for i, member in enumerate(members):
                if member.endswith(".jpg"):
                    # Lấy tên file bỏ qua đường dẫn lồng nhau nếu có
                    filename = Path(member).name
                    target = IMAGES_DIR / filename
                    if not target.exists():
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                if (i + 1) % 3000 == 0:
                    print(f"  Đã giải nén {i + 1}/{len(members)} ảnh...")

    final_count = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"\n=> HOÀN TẤT: Thư mục {IMAGES_DIR} hiện có {final_count} ảnh.")

if __name__ == "__main__":
    setup_data()
