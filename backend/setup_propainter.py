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


def apply_propainter_patches(log=print) -> None:
    target = PROPAINTER_DIR / "inference_propainter.py"
    if not target.is_file():
        return
    text = target.read_text(encoding="utf-8")
    if "_imread_rgb" in text:
        return
    old = """#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name
"""
    new = """#  read frames from video
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _imread_rgb(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def read_frame_from_videos(frame_root):
    video_exts = ('.mp4', '.mov', '.avi', '.m4v', '.mkv', '.webm')
    if os.path.isfile(frame_root) and os.path.splitext(frame_root)[1].lower() in video_exts:
        video_name = os.path.splitext(os.path.basename(frame_root))[0]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(os.path.normpath(frame_root))
        names = sorted(
            n for n in os.listdir(frame_root)
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS
        )
        frames = []
        for fr in names:
            frame = _imread_rgb(os.path.join(frame_root, fr))
            if frame is None:
                continue
            frames.append(frame)
        if not frames:
            raise RuntimeError('No readable image frames in %s' % frame_root)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name
"""
    if old not in text:
        log("未找到可修补的 ProPainter 读帧函数，跳过")
        return
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    log("已修补 ProPainter 读帧逻辑")


def ensure_propainter(log=print) -> Path:
    if propainter_ready():
        apply_propainter_patches(log)
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
    apply_propainter_patches(log)
    return PROPAINTER_DIR


if __name__ == "__main__":
    ensure_propainter()
