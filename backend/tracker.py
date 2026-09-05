from __future__ import annotations

import math
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from ffmpeg_util import run_ffmpeg

SCORE_KEEP = 0.48
MASK_PAD = 12
MAX_PEAKS = 4

Oriented = tuple[int, int, int, int, float]


class JobCancelled(Exception):
    def __init__(self, output_path: str | None = None, engine: str = "delogo"):
        super().__init__("任务已停止")
        self.output_path = output_path
        self.engine = engine


class PauseController:
    def __init__(self) -> None:
        self._can_run = Event()
        self._can_run.set()

    def pause(self) -> None:
        self._can_run.clear()

    def resume(self) -> None:
        self._can_run.set()

    def is_paused(self) -> bool:
        return not self._can_run.is_set()

    def wait_if_paused(self, cancel_event: Event | None = None, on_progress=None) -> None:
        throw_if_cancelled(cancel_event)
        if self._can_run.is_set():
            return
        if on_progress:
            on_progress("paused", None, "已暂停，可点继续")
        while not self._can_run.is_set():
            throw_if_cancelled(cancel_event)
            self._can_run.wait(timeout=0.25)
        throw_if_cancelled(cancel_event)


def throw_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled()


def checkpoint(
    cancel_event: Event | None = None,
    pause: PauseController | None = None,
    on_progress=None,
) -> None:
    throw_if_cancelled(cancel_event)
    if pause is not None:
        pause.wait_if_paused(cancel_event=cancel_event, on_progress=on_progress)


def extract_frames(
    video_path: str,
    out_dir: Path,
    cancel_event: Event | None = None,
    pause: PauseController | None = None,
    on_status=None,
) -> list[Path]:
    checkpoint(cancel_event, pause, on_status)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%06d.jpg")
    result = run_ffmpeg(["-y", "-i", video_path, "-q:v", "2", pattern])
    checkpoint(cancel_event, pause, on_status)
    files = sorted(out_dir.glob("*.jpg"))
    if files:
        return files
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"抽帧失败\n{(result.stderr or '')[-1200:]}")
    index = 1
    while True:
        checkpoint(cancel_event, pause, on_status)
        ok, frame = cap.read()
        if not ok:
            break
        path = out_dir / f"{index:06d}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        index += 1
    cap.release()
    files = sorted(out_dir.glob("*.jpg"))
    if not files:
        raise RuntimeError("未能从视频抽出帧")
    return files


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(x, width - 2))
    y = max(0, min(y, height - 2))
    w = max(8, min(w, width - x))
    h = max(8, min(h, height - y))
    return x, y, w, h


def _crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = _clamp_box(box, frame.shape[1], frame.shape[0])
    return frame[y : y + h, x : x + w]


def oriented_corners(x: float, y: float, w: float, h: float, rotation: float) -> np.ndarray:
    cx = x + w / 2
    cy = y + h / 2
    dx = np.array([x, x + w, x + w, x], dtype=np.float64) - cx
    dy = np.array([y, y, y + h, y + h], dtype=np.float64) - cy
    cos = math.cos(rotation)
    sin = math.sin(rotation)
    return np.stack([cx + dx * cos - dy * sin, cy + dx * sin + dy * cos], axis=1)


def aabb_of(box: Oriented, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h, rotation = box
    pts = oriented_corners(x, y, w, h, rotation)
    min_x = int(np.floor(pts[:, 0].min()))
    min_y = int(np.floor(pts[:, 1].min()))
    max_x = int(np.ceil(pts[:, 0].max()))
    max_y = int(np.ceil(pts[:, 1].max()))
    return _clamp_box((min_x, min_y, max(8, max_x - min_x), max(8, max_y - min_y)), width, height)


def place_oriented(seed: Oriented, found: tuple[int, int, int, int], width: int, height: int) -> Oriented:
    x, y, w, h, rotation = seed
    seed_aabb = aabb_of(seed, width, height)
    dx = (found[0] + found[2] / 2) - (seed_aabb[0] + seed_aabb[2] / 2)
    dy = (found[1] + found[3] / 2) - (seed_aabb[1] + seed_aabb[3] / 2)
    nx = int(round(x + dx))
    ny = int(round(y + dy))
    return (nx, ny, w, h, rotation)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0


def match_peaks(
    frame_gray: np.ndarray,
    template: np.ndarray,
    threshold: float = SCORE_KEEP,
    max_peaks: int = MAX_PEAKS,
) -> list[tuple[int, int, int, int]]:
    th, tw = template.shape[:2]
    fh, fw = frame_gray.shape[:2]
    if fh < th or fw < tw:
        return []
    result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    peaks: list[tuple[int, int, int, int]] = []
    work = result.copy()
    for _ in range(max_peaks):
        _, score, _, loc = cv2.minMaxLoc(work)
        if float(score) < threshold:
            break
        box = _clamp_box((loc[0], loc[1], tw, th), fw, fh)
        if all(_iou(box, existing) < 0.35 for existing in peaks):
            peaks.append(box)
        x, y = loc
        x0 = max(0, x - tw // 2)
        y0 = max(0, y - th // 2)
        work[y0 : y + th, x0 : x + tw] = 0
    return peaks


def track_region(
    frames: list[Path],
    start_index: int,
    seed: Oriented,
    on_progress=None,
    cancel_event: Event | None = None,
    pause: PauseController | None = None,
    on_status=None,
) -> list[list[Oriented]]:
    n = len(frames)
    boxes: list[list[Oriented]] = [[] for _ in range(n)]
    start_index = max(0, min(n - 1, start_index))
    start_bgr = cv2.imread(str(frames[start_index]))
    if start_bgr is None:
        raise RuntimeError("无法读取框选所在帧")
    height, width = start_bgr.shape[:2]
    x, y, w, h, rotation = seed
    seed = (x, y, w, h, float(rotation))
    aabb = aabb_of(seed, width, height)
    template = _gray(_crop(start_bgr, aabb))
    if template.size == 0:
        raise RuntimeError("框选区域太小")
    boxes[start_index] = [seed]

    for idx, frame_path in enumerate(frames):
        checkpoint(cancel_event, pause, on_status)
        if idx == start_index:
            continue
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        found = match_peaks(_gray(bgr), template)
        boxes[idx] = [place_oriented(seed, item, width, height) for item in found]
        if on_progress and idx % 6 == 0:
            on_progress(idx, n)
    boxes[start_index] = [seed]
    others = match_peaks(_gray(start_bgr), template)
    for item in others:
        if _iou(item, aabb) < 0.35:
            boxes[start_index].append(place_oriented(seed, item, width, height))
    return boxes


def write_union_masks(
    frames: list[Path],
    region_boxes: list[list[list[Oriented]]],
    mask_dir: Path,
    pad: int = MASK_PAD,
    cancel_event: Event | None = None,
    pause: PauseController | None = None,
    on_status=None,
) -> Path:
    mask_dir.mkdir(parents=True, exist_ok=True)
    sample = cv2.imread(str(frames[0]))
    if sample is None:
        raise RuntimeError("无法读取帧来生成遮罩")
    height, width = sample.shape[:2]
    k = max(1, pad * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for i, frame_path in enumerate(frames):
        checkpoint(cancel_event, pause, on_status)
        mask = np.zeros((height, width), dtype=np.uint8)
        for per_region in region_boxes:
            frame_boxes = per_region[i] if i < len(per_region) else []
            for box in frame_boxes:
                x, y, w, h, rotation = box
                pts = np.round(oriented_corners(x, y, w, h, rotation)).astype(np.int32)
                cv2.fillConvexPoly(mask, pts, 255)
        if pad > 0:
            mask = cv2.dilate(mask, kernel)
        cv2.imwrite(str(mask_dir / f"{frame_path.stem}.png"), mask)
    return mask_dir


def inpaint_frames(
    frames: list[Path],
    mask_dir: Path,
    out_dir: Path,
    on_progress=None,
    cancel_event: Event | None = None,
    pause: PauseController | None = None,
    on_status=None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, frame_path in enumerate(frames):
        if pause is not None:
            pause.wait_if_paused(cancel_event=None, on_progress=on_status)
        if cancel_event is not None and cancel_event.is_set():
            break
        image = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_dir / f"{frame_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if mask is None or int(mask.max()) == 0:
            out = image
        else:
            out = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        dest = out_dir / frame_path.name
        cv2.imwrite(str(dest), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        written.append(dest)
        if on_progress and i % 4 == 0:
            on_progress(i, len(frames))
    return written


def encode_frames(frame_files: list[Path], fps: float, output_path: str) -> None:
    if not frame_files:
        raise RuntimeError("没有可编码的帧")
    pattern = str(frame_files[0].parent / "%06d.jpg")
    result = run_ffmpeg(
        [
            "-y",
            "-framerate",
            f"{fps:.4f}",
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"合成视频失败\n{(result.stderr or '')[-1500:]}")
