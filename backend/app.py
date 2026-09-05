from __future__ import annotations

import asyncio
import json
import socket
import sys
import traceback
import uuid
from pathlib import Path
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ffmpeg_util import find_ffmpeg
from pipeline import probe_video, run_job
from propainter_runner import cuda_status
from setup_propainter import ensure_propainter, propainter_ready
from tracker import JobCancelled, PauseController

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / ".jobs"

app = FastAPI(title="WaterMark Sidecar")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict[str, Any]] = {}


class Rect(BaseModel):
    x: float
    y: float
    width: float
    height: float
    timeSec: float = 0
    rotation: float = 0


class CreateJob(BaseModel):
    inputPath: str
    rect: Rect | None = None
    regions: list[Rect] = Field(default_factory=list)
    engine: str = Field(default="auto")
    maxEdge: int = Field(default=1280)
    track: bool = True


@app.get("/health")
def health():
    ffmpeg = find_ffmpeg()
    cuda = cuda_status()
    return {
        "ok": True,
        "ffmpeg": ffmpeg,
        "ffmpegReady": bool(ffmpeg),
        "propainterReady": propainter_ready(),
        "cuda": cuda,
        "python": sys.executable,
        "jobsDir": str(JOBS_DIR),
    }


@app.get("/probe")
def probe(path: str):
    if not Path(path).is_file():
        raise HTTPException(400, "视频文件不存在")
    try:
        return probe_video(path)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/jobs")
async def create_job(body: CreateJob):
    input_path = Path(body.inputPath)
    if not input_path.is_file():
        raise HTTPException(400, "视频文件不存在")
    if body.engine not in {"auto", "propainter", "delogo"}:
        raise HTTPException(400, "不支持的引擎")
    if not body.regions and body.rect is None:
        raise HTTPException(400, "请至少框选一个水印区域")
    job_id = uuid.uuid4().hex[:12]
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    pause = PauseController()
    job = {
        "id": job_id,
        "status": "running",
        "queue": queue,
        "outputPath": None,
        "engine": None,
        "error": None,
        "cancel": False,
        "cancel_event": cancel_event,
        "pause": pause,
    }
    jobs[job_id] = job

    async def runner():
        loop = asyncio.get_running_loop()

        def progress(stage: str, pct: float | None, message: str):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"stage": stage, "progress": pct, "message": message},
            )

        try:
            output, engine, partial = await asyncio.to_thread(
                run_job,
                input_path=str(input_path),
                work_dir=work_dir,
                rect=body.rect.model_dump() if body.rect else None,
                regions=[item.model_dump() for item in body.regions],
                engine=body.engine,
                max_edge=body.maxEdge,
                track=body.track,
                on_progress=progress,
                cancel_event=cancel_event,
                pause=pause,
            )
            job["status"] = "done"
            job["outputPath"] = output
            job["engine"] = engine
            await queue.put(
                {
                    "stage": "done",
                    "progress": 1,
                    "message": "部分完成，可查看已处理片段" if partial else "完成",
                    "outputPath": output,
                    "engine": engine,
                    "partial": partial,
                }
            )
        except JobCancelled as exc:
            if exc.output_path:
                job["status"] = "done"
                job["outputPath"] = exc.output_path
                await queue.put(
                    {
                        "stage": "done",
                        "progress": 1,
                        "message": "部分完成，可查看已处理片段",
                        "outputPath": exc.output_path,
                        "engine": exc.engine,
                        "partial": True,
                    }
                )
            else:
                job["status"] = "cancelled"
                await queue.put({"stage": "cancelled", "progress": 0, "message": "已取消，尚未开始修复"})
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            await queue.put({"stage": "error", "message": str(exc), "detail": traceback.format_exc()})
        finally:
            await queue.put(None)

    asyncio.create_task(runner())
    return {"jobId": job_id}


@app.post("/jobs/{job_id}/pause")
def pause_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    controller = job.get("pause")
    if controller is None:
        raise HTTPException(400, "任务不支持暂停")
    job["status"] = "pausing"
    controller.pause()
    return {"ok": True}


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    controller = job.get("pause")
    if controller is None:
        raise HTTPException(400, "任务不支持继续")
    job["status"] = "running"
    controller.resume()
    return {"ok": True}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    job["cancel"] = True
    event = job.get("cancel_event")
    if event is not None:
        event.set()
    controller = job.get("pause")
    if controller is not None:
        controller.resume()
    return {"ok": True}


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    async def gen():
        queue: asyncio.Queue = job["queue"]
        yield ": connected\n\n"
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/setup/propainter")
async def setup_propainter_endpoint():
    logs: list[str] = []

    def log(message: str):
        logs.append(message)

    try:
        await asyncio.to_thread(ensure_propainter, log)
        return {"ok": True, "logs": logs, "ready": propainter_ready()}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


def pick_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    import uvicorn

    port = pick_port()
    print(f"WATERMARK_READY port={port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
