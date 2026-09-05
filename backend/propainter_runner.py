from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from threading import Event

from setup_propainter import PROPAINTER_DIR, propainter_ready
from tracker import JobCancelled

OOM_HINT = (
    "显存不足。请把「处理分辨率」调低后再试，或关闭其他占用 GPU 的程序。"
)


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
) -> Path:
    if not propainter_ready():
        raise RuntimeError("未安装 ProPainter。请先运行 python backend/setup_propainter.py")

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
    combined = []
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
        if "out of memory" in text.lower() or "cuda out of memory" in text.lower():
            raise RuntimeError(OOM_HINT)
        raise RuntimeError(f"ProPainter 失败（退出码 {proc.returncode}）\n{text[-2000:]}")

    out = Path(output_dir)
    candidates = list(out.rglob("inpaint_out.mp4"))
    if not candidates:
        raise RuntimeError("ProPainter 未生成 inpaint_out.mp4")
    return candidates[0]
