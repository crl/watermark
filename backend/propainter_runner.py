from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Event

import cv2

from setup_propainter import PROPAINTER_DIR, propainter_ready
from tracker import JobCancelled, PauseController, checkpoint, encode_frames

OOM_HINT = (
    "显存不足。请把「处理分辨率」调低后再试，或关闭其他占用 GPU 的程序。"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cuda_status() -> dict:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if available else None
        return {
            "torch": True,
            "cuda": available,
            "device": name,
            "version": getattr(torch, "__version__", None),
        }
    except Exception:
        return {"torch": False, "cuda": False, "device": None, "version": None}


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _run_once(
    video_path: str,
    mask_path: str,
    output_dir: str,
    *,
    resize_ratio: float,
    fp16: bool,
    save_fps: int,
    subvideo_length: int,
    on_log,
    cancel_event: Event | None,
) -> Path:
    cmd = [
        sys.executable,
        "inference_propainter.py",
        "--video",
        video_path,
        "--mask",
        mask_path,
        "--output",
        output_dir,
        "--resize_ratio",
        str(resize_ratio),
        "--save_fps",
        str(max(1, save_fps)),
        "--subvideo_length",
        str(subvideo_length),
        "--save_frames",
    ]
    if fp16:
        cmd.append("--fp16")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROPAINTER_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    combined: list[str] = []
    progress_re = re.compile(r"(\d+)%\|")
    try:
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                break
            combined.append(line)
            stripped = line.strip()
            if stripped:
                on_log(stripped)
            match = progress_re.search(line)
            if match:
                on_log(f"PROPAINTER_PCT {match.group(1)}")
    finally:
        proc.wait()

    if cancel_event is not None and cancel_event.is_set():
        out = Path(output_dir)
        candidates = list(out.rglob("inpaint_out.mp4"))
        if candidates:
            return candidates[0]
        raise JobCancelled(None, "propainter")

    text = "".join(combined)
    if proc.returncode != 0:
        lower = text.lower()
        if "out of memory" in lower or "cuda out of memory" in lower or "memoryerror" in lower:
            raise RuntimeError(OOM_HINT)
        raise RuntimeError(f"ProPainter 失败（退出码 {proc.returncode}）\n{text[-2000:]}")

    out = Path(output_dir)
    candidates = list(out.rglob("inpaint_out.mp4"))
    if not candidates:
        raise RuntimeError("ProPainter 未生成 inpaint_out.mp4")
    return candidates[0]


def _chunk_ranges(count: int, size: int, overlap: int) -> list[tuple[int, int]]:
    size = max(8, size)
    overlap = max(0, min(overlap, size // 2))
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(count, start + size)
        ranges.append((start, end))
        if end >= count:
            break
        start = end - overlap
    return ranges


def _collect_chunk_frames(chunk_output: Path) -> list[Path]:
    frame_dirs = list(chunk_output.rglob("frames"))
    for folder in frame_dirs:
        images = _list_images(folder)
        if images:
            return images
    return []


def _run_chunked(
    frames_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    *,
    resize_ratio: float,
    fp16: bool,
    save_fps: int,
    subvideo_length: int,
    on_log,
    cancel_event: Event | None,
    pause: PauseController | None = None,
    on_status=None,
) -> Path:
    frames = _list_images(frames_dir)
    if not frames:
        raise RuntimeError("没有可送入 ProPainter 的帧")
    overlap = min(8, max(2, subvideo_length // 8))
    windows = _chunk_ranges(len(frames), subvideo_length, overlap)
    on_log(f"视频较长（{len(frames)} 帧），将分 {len(windows)} 段修复")
    stitched = output_dir / "stitched"
    stitched.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, (start, end) in enumerate(windows, start=1):
        if pause is not None:
            pause.wait_if_paused(cancel_event=None, on_progress=on_status)
        if cancel_event is not None and cancel_event.is_set():
            break
        chunk_root = output_dir / "chunks" / f"{index:04d}"
        chunk_frames = chunk_root / "in"
        chunk_masks = chunk_root / "masks"
        chunk_out = chunk_root / "out"
        slice_frames = frames[start:end]
        for src in slice_frames:
            _link_or_copy(src, chunk_frames / src.name)
            mask_src = masks_dir / f"{src.stem}.png"
            if mask_src.exists():
                _link_or_copy(mask_src, chunk_masks / mask_src.name)
        on_log(f"PROPAINTER_PCT {int((index - 1) / max(len(windows), 1) * 100)}")
        on_log(f"ProPainter 第 {index}/{len(windows)} 段（帧 {start + 1}-{end}）")
        _run_once(
            str(chunk_frames),
            str(chunk_masks),
            str(chunk_out),
            resize_ratio=resize_ratio,
            fp16=fp16,
            save_fps=save_fps,
            subvideo_length=subvideo_length,
            on_log=on_log,
            cancel_event=cancel_event,
        )
        produced = _collect_chunk_frames(chunk_out)
        if not produced:
            raise RuntimeError(f"第 {index} 段未产出修复帧")
        skip = 0 if index == 1 else min(overlap, len(produced) - 1)
        for src in produced[skip:]:
            written += 1
            dest = stitched / f"{written:06d}.jpg"
            image = cv2.imread(str(src))
            if image is None:
                continue
            cv2.imwrite(str(dest), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        shutil.rmtree(chunk_root, ignore_errors=True)

    stitched_files = _list_images(stitched)
    if not stitched_files:
        raise JobCancelled(None, "propainter") if cancel_event is not None and cancel_event.is_set() else RuntimeError(
            "分段修复后没有可用画面"
        )
    visual = output_dir / "inpaint_out.mp4"
    encode_frames(stitched_files, float(max(1, save_fps)), str(visual))
    return visual


def run_propainter(
    video_path: str,
    mask_path: str,
    output_dir: str,
    *,
    resize_ratio: float,
    fp16: bool,
    save_fps: int,
    subvideo_length: int,
    on_log,
    cancel_event: Event | None = None,
    pause: PauseController | None = None,
    on_status=None,
) -> Path:
    if not propainter_ready():
        raise RuntimeError("未安装 ProPainter。请先运行 python backend/setup_propainter.py")

    source = Path(video_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if pause is not None:
        pause.wait_if_paused(cancel_event=None, on_progress=on_status)
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled(None, "propainter")
    if source.is_dir() and len(_list_images(source)) > subvideo_length:
        return _run_chunked(
            source,
            Path(mask_path),
            output,
            resize_ratio=resize_ratio,
            fp16=fp16,
            save_fps=save_fps,
            subvideo_length=subvideo_length,
            on_log=on_log,
            cancel_event=cancel_event,
            pause=pause,
            on_status=on_status,
        )
    return _run_once(
        video_path,
        mask_path,
        output_dir,
        resize_ratio=resize_ratio,
        fp16=fp16,
        save_fps=save_fps,
        subvideo_length=subvideo_length,
        on_log=on_log,
        cancel_event=cancel_event,
    )
