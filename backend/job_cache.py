from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / ".jobs"
CACHE_DIR = JOBS_DIR / "cache"
FRAMES_CACHE = CACHE_DIR / "frames"
RESULTS_CACHE = CACHE_DIR / "results"
USAGE_FILE = CACHE_DIR / "usage.json"

_lock = threading.Lock()


def file_fingerprint(path: str | Path) -> str:
    target = Path(path).resolve()
    stat = target.stat()
    raw = f"{target}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _canonical_regions(regions: list[dict] | None, rect: dict | None) -> list[dict]:
    items = list(regions or [])
    if not items and rect:
        items = [rect]
    canon = []
    for item in items:
        canon.append(
            {
                "x": round(float(item.get("x") or 0), 2),
                "y": round(float(item.get("y") or 0), 2),
                "width": round(float(item.get("width") or 0), 2),
                "height": round(float(item.get("height") or 0), 2),
                "rotation": round(float(item.get("rotation") or 0), 4),
                "timeSec": round(float(item.get("timeSec") or 0), 3),
            }
        )
    canon.sort(key=lambda item: (item["timeSec"], item["x"], item["y"], item["width"], item["height"], item["rotation"]))
    return canon


def frames_key(fingerprint: str, max_edge: int) -> str:
    return _digest({"file": fingerprint, "maxEdge": int(max_edge)})


def result_key(
    fingerprint: str,
    max_edge: int,
    engine: str,
    track: bool,
    regions: list[dict] | None,
    rect: dict | None,
) -> str:
    return _digest(
        {
            "file": fingerprint,
            "maxEdge": int(max_edge),
            "engine": engine,
            "track": bool(track),
            "regions": _canonical_regions(regions, rect),
        }
    )


def _list_jpgs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _copy_tree(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_file in _list_jpgs(src):
        _link_or_copy(src_file, dest / src_file.name)
        count += 1
    return count


def find_frames(fingerprint: str, max_edge: int) -> Path | None:
    key = frames_key(fingerprint, max_edge)
    folder = FRAMES_CACHE / key
    with _lock:
        files = _list_jpgs(folder)
        return folder if files else None


def store_frames(src: Path, fingerprint: str, max_edge: int) -> None:
    files = _list_jpgs(src)
    if not files:
        return
    dest = FRAMES_CACHE / frames_key(fingerprint, max_edge)
    with _lock:
        if _list_jpgs(dest):
            return
        added = _copy_tree(src, dest)
        if added:
            _add_usage(_dir_size(dest))


def materialize_frames(cached: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    _copy_tree(cached, dest)
    files = _list_jpgs(dest)
    if not files:
        raise RuntimeError("缓存帧无法写入工作目录")
    return files


def find_result(key: str) -> dict | None:
    folder = RESULTS_CACHE / key
    output = folder / "output.mp4"
    meta_path = folder / "meta.json"
    with _lock:
        if not output.is_file() or output.stat().st_size <= 0:
            return None
        engine = None
        if meta_path.is_file():
            try:
                engine = json.loads(meta_path.read_text(encoding="utf-8")).get("engine")
            except json.JSONDecodeError:
                engine = None
        return {"outputPath": str(output), "engine": engine, "folder": str(folder)}


def store_result(output_path: str | Path, key: str, engine: str) -> None:
    src = Path(output_path)
    if not src.is_file() or src.stat().st_size <= 0:
        return
    dest_dir = RESULTS_CACHE / key
    dest = dest_dir / "output.mp4"
    with _lock:
        dest_dir.mkdir(parents=True, exist_ok=True)
        created = not dest.exists()
        if created:
            _link_or_copy(src, dest)
            try:
                _add_usage(dest.stat().st_size)
            except OSError:
                pass
        dest_dir.joinpath("meta.json").write_text(
            json.dumps({"engine": engine}, ensure_ascii=False),
            encoding="utf-8",
        )


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _read_usage() -> int:
    if not USAGE_FILE.is_file():
        return 0
    try:
        return max(0, int(json.loads(USAGE_FILE.read_text(encoding="utf-8")).get("bytes") or 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _write_usage(size: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps({"bytes": max(0, int(size))}), encoding="utf-8")


def _add_usage(delta: int) -> None:
    if not delta:
        return
    _write_usage(_read_usage() + delta)


def format_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            text = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{text} {unit}"
        value /= 1024
    return f"{size} B"


def cache_stats() -> dict:
    jobs = (
        [path for path in JOBS_DIR.iterdir() if path.is_dir() and path.name not in {"cache"}]
        if JOBS_DIR.exists()
        else []
    )
    frames = [path for path in FRAMES_CACHE.iterdir() if path.is_dir()] if FRAMES_CACHE.exists() else []
    results = [path for path in RESULTS_CACHE.iterdir() if path.is_dir()] if RESULTS_CACHE.exists() else []
    bytes_used = _read_usage()
    if bytes_used > 0:
        label = format_bytes(bytes_used)
    elif jobs or frames or results:
        label = f"{len(jobs)} 个任务"
        bytes_used = max(len(jobs) + len(frames) + len(results), 1)
    else:
        label = "空"
    return {
        "bytes": bytes_used,
        "label": label,
        "jobs": len(jobs),
        "frames": len(frames),
        "results": len(results),
    }


def _rmtree(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        shutil.rmtree(path)
        return True
    except OSError:
        return False


def prune_work_dir(work_dir: Path) -> None:
    for name in ("frames", "masks", "propainter", "repaired"):
        _rmtree(work_dir / name)
    visual = work_dir / "visual.mp4"
    if visual.exists():
        try:
            visual.unlink()
        except OSError:
            pass


def clear_cache(keep_job_ids: set[str] | None = None, on_progress=None) -> dict:
    keep = set(keep_job_ids or ())
    targets: list[Path] = []
    if JOBS_DIR.exists():
        for path in list(JOBS_DIR.iterdir()):
            if path.name == "cache":
                continue
            if path.is_dir() and path.name in keep:
                continue
            targets.append(path)
    if CACHE_DIR.exists():
        targets.append(CACHE_DIR)

    total = max(len(targets), 1)
    removed = 0
    before_jobs = len(targets)
    for index, path in enumerate(targets):
        if on_progress:
            on_progress(index, total, path.name)
        if path.is_dir():
            if _rmtree(path):
                removed += 1
        else:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    if on_progress:
        on_progress(total, total, "完成")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        _write_usage(0)
    stats = cache_stats()
    stats["removed"] = removed
    stats["freed"] = 0
    stats["freedLabel"] = f"{before_jobs} 个任务目录" if before_jobs else "缓存"
    return stats
