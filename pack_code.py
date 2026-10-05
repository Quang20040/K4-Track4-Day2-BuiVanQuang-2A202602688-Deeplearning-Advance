import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_OUT = ROOT / "code_colab.zip"

files_to_pack = [
    ROOT / "eval.py",
    ROOT / "run_all_experiments.py",
    ROOT / "prepare_data.py",
]

starter_files = list((ROOT / "starter").glob("*.py"))

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files_to_pack:
        if f.exists():
            zf.write(f, arcname=f.name)
            print(f"Đã thêm: {f.name}")
    for f in starter_files:
        if f.exists():
            zf.write(f, arcname=f"starter/{f.name}")
            print(f"Đã thêm: starter/{f.name}")

print(f"\n=> ĐÃ TẠO XONG: {ZIP_OUT} ({ZIP_OUT.stat().st_size / 1024:.1f} KB)")
