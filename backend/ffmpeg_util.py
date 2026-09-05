from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_ffmpeg() -> str | None:
    env = os.environ.get("WATERMARK_FFMPEG")
    if env and Path(env).exists():
        return env

    bundled = Path(__file__).resolve().parent.parent / "resources" / "ffmpeg" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)

    which = shutil.which("ffmpeg")
    if which:
        return which

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_ffmpeg(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未找到 FFmpeg。请安装 FFmpeg 或 pip install imageio-ffmpeg")
    cmd = [ffmpeg, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def has_audio_stream(video_path: str) -> bool:
    result = run_ffmpeg(["-hide_banner", "-i", video_path])
    text = (result.stderr or "") + (result.stdout or "")
    return "Audio:" in text
