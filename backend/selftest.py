from __future__ import annotations

from pathlib import Path

from ffmpeg_util import find_ffmpeg, run_ffmpeg
from pipeline import run_job


def main() -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg missing")
    root = Path(__file__).resolve().parent / ".jobs" / "selftest"
    root.mkdir(parents=True, exist_ok=True)
    src = root / "input.mp4"
    result = run_ffmpeg(
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=320x240:d=1:r=12",
            "-vf",
            "drawbox=x=10:y=20:w=48:h=20:color=white:t=fill,drawbox=x=250:y=190:w=50:h=18:color=yellow:t=fill",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ]
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr[-800:])
    output, engine = run_job(
        input_path=str(src),
        work_dir=root / "out",
        rect=None,
        regions=[
            {"x": 10, "y": 20, "width": 48, "height": 20, "timeSec": 0},
            {"x": 250, "y": 190, "width": 50, "height": 18, "timeSec": 0},
        ],
        engine="delogo",
        max_edge=720,
        track=True,
        on_progress=lambda stage, pct, message: print(stage, pct, message),
    )[:2]
    if not Path(output).exists():
        raise SystemExit("output missing")
    print("OK", engine, output)


if __name__ == "__main__":
    main()
