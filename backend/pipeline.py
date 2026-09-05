from __future__ import annotations

import shutil
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from PIL import Image

from ffmpeg_util import find_ffmpeg, has_audio_stream, run_ffmpeg
from propainter_runner import cuda_status, run_propainter
from setup_propainter import propainter_ready
from tracker import (
    JobCancelled,
    encode_frames,
    extract_frames,
    inpaint_frames,
    throw_if_cancelled,
    track_region,
    write_union_masks,
)


def probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频，请确认文件未损坏且格式受支持")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("读取视频尺寸失败")
    duration = frames / fps if fps > 0 and frames > 0 else 0
    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 1 else 24.0,
        "frames": frames,
        "duration": duration,
        "hasAudio": has_audio_stream(path),
    }


def clamp_rect(rect: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x = int(round(rect["x"]))
    y = int(round(rect["y"]))
    w = int(round(rect["width"]))
    h = int(round(rect["height"]))
    x = max(0, min(x, width - 2))
    y = max(0, min(y, height - 2))
    w = max(8, min(w, width - x))
    h = max(8, min(h, height - y))
    return x, y, w, h


def clamp_oriented(rect: dict, width: int, height: int) -> tuple[int, int, int, int, float]:
    x, y, w, h = clamp_rect(rect, width, height)
    return x, y, w, h, float(rect.get("rotation") or 0)


def compute_resize_ratio(width: int, height: int, max_edge: int) -> float:
    longest = max(width, height)
    if max_edge <= 0 or longest <= max_edge:
        return 1.0
    return max_edge / longest


def write_mask(path: Path, width: int, height: int, rects: list[tuple[int, int, int, int]]) -> None:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in rects:
        mask[y : y + h, x : x + w] = 255
    Image.fromarray(mask, mode="L").save(path)


def mux_audio(visual: str, original: str, output: str, fps: float) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copy2(visual, output)
        return
    args = ["-y", "-i", visual, "-i", original, "-map", "0:v:0"]
    if has_audio_stream(original):
        args += ["-map", "1:a:0?", "-c:a", "copy"]
    args += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-r",
        f"{fps:.4f}",
        "-movflags",
        "+faststart",
        "-shortest",
        output,
    ]
    result = run_ffmpeg(args)
    if result.returncode != 0 or not Path(output).exists():
        raise RuntimeError(f"合成音轨失败\n{(result.stderr or '')[-1500:]}")


def choose_engine(engine: str) -> str:
    if engine == "delogo":
        return "delogo"
    if engine == "propainter":
        if not propainter_ready():
            raise RuntimeError("未安装 ProPainter。请先在应用内下载源码，或运行 python backend/setup_propainter.py")
        status = cuda_status()
        if not status.get("torch"):
            raise RuntimeError(
                "未安装 PyTorch。请先安装 CUDA 版 PyTorch，再 pip install -r backend/requirements-ai.txt"
            )
        return "propainter"
    status = cuda_status()
    if propainter_ready() and status.get("torch"):
        return "propainter"
    return "delogo"


def _normalize_regions(regions: list[dict] | None, rect: dict | None) -> list[dict]:
    items = list(regions or [])
    if not items and rect:
        items = [{**rect, "timeSec": float(rect.get("timeSec") or 0)}]
    if not items:
        raise RuntimeError("请至少框选一个水印区域")
    return items


def _finish_visual(
    visual: str,
    input_path: str,
    output_path: str,
    fps: float,
    on_progress,
    partial: bool,
    used_engine: str,
) -> tuple[str, str, bool]:
    on_progress("mux", 0.9, "合并原音轨")
    mux_audio(visual, input_path, output_path, fps)
    on_progress("done", 1.0, "部分完成，可查看已处理片段" if partial else "完成")
    return output_path, used_engine, partial


def _track_all(
    frames: list[Path],
    regions: list[dict],
    info: dict,
    on_progress,
    cancel_event: Event | None,
) -> list:
    fps = info["fps"]
    n = len(frames)
    tracked = []
    for index, region in enumerate(regions, start=1):
        throw_if_cancelled(cancel_event)
        seed = clamp_oriented(region, info["width"], info["height"])
        time_sec = float(region.get("timeSec") or 0)
        start_index = int(round(time_sec * fps))
        start_index = max(0, min(n - 1, start_index))
        on_progress("track", 0.08 + 0.18 * ((index - 1) / max(len(regions), 1)), f"全图匹配第 {index}/{len(regions)} 块")
        tracked.append(
            track_region(
                frames,
                start_index,
                seed,
                on_progress=lambda idx, total, i=index, count=len(regions): on_progress(
                    "track",
                    0.08 + 0.2 * ((i - 1 + idx / max(total, 1)) / max(count, 1)),
                    f"匹配第 {i}/{count} 块 {idx + 1}/{total}",
                ),
                cancel_event=cancel_event,
            )
        )
    return tracked


def run_job(
    *,
    input_path: str,
    work_dir: Path,
    rect: dict | None,
    regions: list[dict] | None,
    engine: str,
    max_edge: int,
    track: bool,
    on_progress,
    cancel_event: Event | None = None,
) -> tuple[str, str, bool]:
    work_dir.mkdir(parents=True, exist_ok=True)
    on_progress("probe", 0.04, "读取视频信息")
    info = probe_video(input_path)
    items = _normalize_regions(regions, rect)
    used_engine = choose_engine(engine)
    output_path = str(work_dir / "output.mp4")
    seeds = [clamp_oriented(item, info["width"], info["height"]) for item in items]

    on_progress("extract", 0.06, "抽出视频帧")
    frames = extract_frames(input_path, work_dir / "frames", cancel_event=cancel_event)
    if track:
        region_boxes = _track_all(frames, items, info, on_progress, cancel_event)
    else:
        region_boxes = [[[seed] for _ in frames] for seed in seeds]

    on_progress("mask", 0.30, "生成逐帧遮罩")
    mask_dir = write_union_masks(frames, region_boxes, work_dir / "masks", cancel_event=cancel_event)

    if used_engine == "delogo":
        on_progress("inpaint", 0.35, "逐帧修补")
        repaired = inpaint_frames(
            frames,
            mask_dir,
            work_dir / "repaired",
            on_progress=lambda i, total: on_progress(
                "inpaint",
                0.35 + 0.5 * (i / max(total, 1)),
                f"逐帧修补 {i + 1}/{total}",
            ),
            cancel_event=cancel_event,
        )
        partial = bool(cancel_event and cancel_event.is_set())
        if not repaired:
            raise JobCancelled(None, used_engine)
        visual = str(work_dir / "visual.mp4")
        encode_frames(repaired, info["fps"], visual)
        return _finish_visual(visual, input_path, output_path, info["fps"], on_progress, partial, used_engine)

    ratio = compute_resize_ratio(info["width"], info["height"], max_edge)
    on_progress("inpaint", 0.34, "ProPainter 修复中")
    pp_out_dir = str(work_dir / "propainter")
    Path(pp_out_dir).mkdir(parents=True, exist_ok=True)
    last_error = None
    visual = None
    for attempt_ratio in (ratio, min(ratio, 0.5), min(ratio, 0.35)):
        throw_if_cancelled(cancel_event)
        try:

            def log_line(line: str):
                if line.startswith("PROPAINTER_PCT "):
                    pct = int(line.split()[1]) / 100
                    on_progress("inpaint", 0.35 + pct * 0.5, f"ProPainter 修复中 {int(pct * 100)}%")
                elif "Processing:" in line or "saved in" in line:
                    on_progress("inpaint", None, line[:180])

            visual = run_propainter(
                str(work_dir / "frames"),
                str(mask_dir),
                pp_out_dir,
                resize_ratio=attempt_ratio,
                fp16=True,
                save_fps=int(round(info["fps"])),
                subvideo_length=80 if attempt_ratio >= 0.7 else 40,
                on_log=log_line,
                cancel_event=cancel_event,
            )
            last_error = None
            break
        except JobCancelled:
            raise
        except RuntimeError as exc:
            last_error = exc
            if "显存不足" in str(exc) and attempt_ratio > 0.36:
                on_progress("inpaint", 0.35, "显存不足，降低分辨率重试")
                continue
            raise

    if last_error:
        raise last_error
    if visual is None:
        raise RuntimeError("ProPainter 未产出结果")

    return _finish_visual(str(visual), input_path, output_path, info["fps"], on_progress, False, used_engine)
