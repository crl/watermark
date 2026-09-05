from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
PROPAINTER_DIR = VENDOR_DIR / "ProPainter"
ARCHIVE_URL = "https://github.com/sczhou/ProPainter/archive/refs/heads/main.zip"


def propainter_ready() -> bool:
    return (PROPAINTER_DIR / "inference_propainter.py").is_file()


def ensure_propainter(log=print) -> Path:
    if propainter_ready():
        log("ProPainter 已就绪")
        return PROPAINTER_DIR

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = VENDOR_DIR / "ProPainter-main.zip"
    log("正在下载 ProPainter 源码…")
    urlretrieve(ARCHIVE_URL, zip_path)
    log("正在解压…")
    extract_to = VENDOR_DIR / "_extract"
    if extract_to.exists():
        shutil.rmtree(extract_to, ignore_errors=True)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    extracted = next(extract_to.iterdir())
    if PROPAINTER_DIR.exists():
        shutil.rmtree(PROPAINTER_DIR, ignore_errors=True)
    shutil.move(str(extracted), str(PROPAINTER_DIR))
    shutil.rmtree(extract_to, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    if not propainter_ready():
        raise RuntimeError("ProPainter 解压后未找到 inference_propainter.py")
    log("ProPainter 源码安装完成")
    return PROPAINTER_DIR


if __name__ == "__main__":
    ensure_propainter()
