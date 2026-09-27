#!/usr/bin/env python3
"""FastAPI server: upload images and stream the stitching progress live.

Run::

    python server.py            # http://127.0.0.1:8000
    python server.py --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from stitcher.grid import DEFAULT_TEMPLATE, NAME_HELP, inspect_names, require_positions

import cv2
import numpy as np
import uvicorn
from fastapi import (Body, FastAPI, File, Form, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from stitcher import (PanoramaStitcher, IncrementalStitcher, StitchConfig,
                      StitchError, ProgressReporter, INCR_STAGES)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
PROJECTS_DIR = BASE_DIR / "projects"
INCR_DIR = BASE_DIR / "incremental"

from comparison import router as comparison_router

app = FastAPI(title="AI Panorama Stitcher", version="1.0.0")
app.include_router(comparison_router)
from microscope_api import router as microscope_router
app.include_router(microscope_router)
from guided_stitching import router as guided_router
app.include_router(guided_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Never cache the UI/static assets (the app changes frequently)."""
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/compare", "/classic", "/microscope", "/guided", "/index.html") or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


jobs: dict[str, "Job"] = {}
incremental_sessions: dict[str, "IncrementalSession"] = {}

jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()


class Job:
    def __init__(self, loop):
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.events = []
        self.lock = threading.Lock()
        self.seq = 0
        self.done = False
        self.result_path = None
        self.upload_dir = None
        self.config = {}
        self.num_images = 0
        self.created_at = datetime.now(timezone.utc).isoformat()

    def emit(self, event):
        with self.lock:
            self.seq += 1
            event = dict(event)
            event["seq"] = self.seq
            self.events.append(event)
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:
            pass

    def history(self):
        with self.lock:
            return list(self.events)


def _run_job(job_id: str, paths, config_dict):
    job = jobs[job_id]
    reporter = ProgressReporter(emit=job.emit)
    cfg = StitchConfig(**config_dict)
    stitcher = PanoramaStitcher(cfg, reporter)
    try:
        result = stitcher.stitch_paths(paths)
        job.result_path = Path(job.upload_dir) / "panorama.jpg"
        cv2.imwrite(str(job.result_path), result.panorama,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    except Exception as exc:  # noqa: BLE001
        reporter.error(str(exc))
    finally:
        job.done = True
        job.emit({"type": "end"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "overview.html")


@app.get("/compare")
def comparison_page():
    return FileResponse(STATIC_DIR / "compare.html")


@app.get("/classic")
def classic_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "stitch-lab", "overview": "/", "jobs": len(jobs),
            "incremental": len(incremental_sessions)}


def _stitch_config(engine, weights, max_keypoints, match_threshold,
                   feature_max_dim, sinkhorn_iterations, ransac_thresh,
                   reference, refine, blend_levels, exposure, crop, viz=True,
                   min_extend_ratio=0.0, min_extend_px=0.0, alignment="homography"):
    return StitchConfig(
        engine=engine,
        alignment=alignment,
        superglue_weights=weights,
        max_keypoints=int(max_keypoints),
        match_threshold=float(match_threshold),
        feature_max_dim=int(feature_max_dim),
        sinkhorn_iterations=int(sinkhorn_iterations),
        ransac_thresh=float(ransac_thresh),
        reference=reference,
        refine=bool(refine),
        blend_levels=int(blend_levels),
        exposure=bool(exposure),
        crop=bool(crop),
        min_extend_ratio=float(min_extend_ratio),
        min_extend_px=float(min_extend_px),
        weights_dir=str(BASE_DIR / "weights"),
        viz=viz,
    )


def _read_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise StitchError("Could not decode the uploaded image.")
    return img


@app.post("/api/grid/positions")
def grid_positions(payload: dict = Body(...)):
    try:
        names = payload.get("names", [])
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("Provide a list of image filenames.")
        result = inspect_names(names, payload.get("template", DEFAULT_TEMPLATE))
        return {**result, "help": NAME_HELP}
    except ValueError as exc:
        return JSONResponse({"error": str(exc), "help": NAME_HELP}, status_code=400)


@app.post("/api/stitch")
async def stitch(
    files: list[UploadFile] = File(...),
    filename_template: str = Form(DEFAULT_TEMPLATE),
    engine: str = Form("superglue"),
    weights: str = Form("outdoor"),
    max_keypoints: int = Form(1024),
    match_threshold: float = Form(0.2),
    feature_max_dim: int = Form(1024),
    sinkhorn_iterations: int = Form(50),
    alignment: Literal["homography", "translation"] = Form("homography"),
    ransac_thresh: float = Form(3.0),
    reference: str = Form("middle"),
    refine: bool = Form(False),
    blend_levels: int = Form(6),
    exposure: bool = Form(True),
    crop: bool = Form(True),
):
    if len(files) < 2:
        return JSONResponse({"error": "Upload at least 2 images."}, status_code=400)

    try:
        positions = require_positions([f.filename or "" for f in files], filename_template)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for f in files:
        ext = Path(f.filename or "img.jpg").suffix.lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
            ext = ".jpg"
        dest = job_dir / f"in_{len(paths):02d}{ext}"
        dest.write_bytes(await f.read())
        paths.append(str(dest))

    loop = asyncio.get_running_loop()
    job = Job(loop)
    job.upload_dir = str(job_dir)

    cfg = _stitch_config(engine, weights, max_keypoints, match_threshold,
                         feature_max_dim, sinkhorn_iterations, ransac_thresh,
                         reference, refine, blend_levels, exposure, crop, alignment=alignment)
    cfg.pairing = "grid"
    cfg.grid_positions = positions
    cfg.input_names = [f.filename for f in files]
    cfg.disconnected = "reject"
    config_dict = cfg.to_dict()
    job.config = dict(config_dict)
    job.config["filename_template"] = filename_template
    job.num_images = len(paths)

    with _jobs_lock:
        jobs[job_id] = job
        _prune_jobs()

    threading.Thread(target=_run_job, args=(job_id, paths, config_dict),
                     daemon=True).start()
    return {"job_id": job_id, "num_images": len(paths)}


@app.websocket("/ws/{job_id}")
async def ws_progress(ws: WebSocket, job_id: str):
    await ws.accept()
    job = jobs.get(job_id)
    if job is None:
        await ws.send_json({"type": "error", "message": "Unknown job id"})
        await ws.close()
        return

    # Replay history (also covers page reloads mid-job).
    last_seq = 0
    for e in job.history():
        if e.get("type") == "end":
            await ws.send_json(e)
            await ws.close()
            return
        await ws.send_json(e)
        last_seq = e.get("seq", 0)

    try:
        while True:
            try:
                e = await asyncio.wait_for(job.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if job.done and job.queue.empty():
                    break
                continue
            if e.get("seq", 0) <= last_seq:
                continue
            if e.get("type") == "end":
                await ws.send_json(e)
                break
            await ws.send_json(e)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


@app.get("/api/jobs/{job_id}")
def job_summary(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return {"error": "Unknown job id"}, 404
    return {"job_id": job_id, "done": job.done,
            "events": len(job.events),
            "result": f"/api/jobs/{job_id}/result" if job.result_path else None}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.result_path is None:
        return {"error": "No result available"}, 404
    return FileResponse(job.result_path, media_type="image/jpeg",
                        filename="panorama.jpg")


# --------------------------------------------------------------------------- #
# Projects (persisted sessions)
# --------------------------------------------------------------------------- #
def _project_thumbnail(pid: str) -> str | None:
    base = PROJECTS_DIR / pid
    if (base / "result.jpg").exists():
        return f"/api/projects/{pid}/files/result.jpg"
    shots = base / "shots"
    if shots.is_dir():
        files = sorted(shots.glob("*.jpg"))
        if files:
            return f"/api/projects/{pid}/files/shots/{files[0].name}"
    return None


def _project_summary(pid: str) -> dict:
    meta = json.loads((PROJECTS_DIR / pid / "meta.json").read_text())
    return {**meta, "thumbnail": _project_thumbnail(pid)}


def _save_project(job_id: str, name: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise KeyError(job_id)

    pid = uuid.uuid4().hex[:12]
    base = PROJECTS_DIR / pid
    (base / "inputs").mkdir(parents=True, exist_ok=True)
    (base / "shots").mkdir(parents=True, exist_ok=True)

    # Copy original input images.
    if job.upload_dir:
        for f in sorted(Path(job.upload_dir).glob("in_*")):
            shutil.copy(f, base / "inputs" / f.name)

    # Copy final panorama (if any).
    result_rel = None
    if job.result_path and Path(job.result_path).exists():
        shutil.copy(job.result_path, base / "result.jpg")
        result_rel = "result.jpg"

    # Persist the event history, extracting base64 images to files.
    n = 0
    with open(base / "events.jsonl", "w") as fh:
        for ev in job.history():
            ev = dict(ev)
            if ev.get("type") == "image" and ev.get("image", "").startswith("data:"):
                try:
                    raw = base64.b64decode(ev["image"].split(",", 1)[1])
                    shot = f"shots/s{n:04d}.jpg"
                    (base / shot).write_bytes(raw)
                    ev["image"] = shot
                    n += 1
                except Exception:  # noqa: BLE001
                    ev["image"] = None
            elif ev.get("type") == "result":
                ev["image"] = result_rel
            fh.write(json.dumps(ev) + "\n")

    meta = {
        "id": pid,
        "name": name or f"Panorama {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "done" if job.done else "running",
        "engine": job.config.get("engine", "superglue"),
        "weights": job.config.get("superglue_weights", ""),
        "num_images": job.num_images,
        "options": job.config,
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2))
    return _project_summary(pid)


@app.get("/api/projects")
def list_projects():
    PROJECTS_DIR.mkdir(exist_ok=True)
    out = []
    for m in PROJECTS_DIR.glob("*/meta.json"):
        try:
            out.append(_project_summary(m.parent.name))
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"projects": out}


@app.post("/api/projects")
def create_project(payload: dict = Body(...)):
    job_id = payload.get("job_id")
    name = (payload.get("name") or "").strip()
    try:
        summary = _save_project(job_id, name)
    except KeyError:
        return JSONResponse({"error": "Unknown job id"}, status_code=404)
    return summary


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    base = PROJECTS_DIR / pid
    meta_path = base / "meta.json"
    if not meta_path.exists():
        return JSONResponse({"error": "Unknown project"}, status_code=404)
    meta = json.loads(meta_path.read_text())
    events = []
    events_path = base / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            img = ev.get("image")
            if img and not img.startswith(("data:", "/api", "http")):
                ev["image"] = f"/api/projects/{pid}/files/{img}"
            events.append(ev)
    return {"meta": meta, "events": events}


@app.post("/api/projects/{pid}/rename")
def rename_project(pid: str, payload: dict = Body(...)):
    meta_path = PROJECTS_DIR / pid / "meta.json"
    if not meta_path.exists():
        return JSONResponse({"error": "Unknown project"}, status_code=404)
    meta = json.loads(meta_path.read_text())
    meta["name"] = (payload.get("name") or meta.get("name") or "Untitled").strip()
    meta_path.write_text(json.dumps(meta, indent=2))
    return _project_summary(pid)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    base = PROJECTS_DIR / pid
    if not base.exists():
        return JSONResponse({"error": "Unknown project"}, status_code=404)
    shutil.rmtree(base, ignore_errors=True)
    return {"deleted": pid}


@app.get("/api/projects/{pid}/files/{fpath:path}")
def project_file(pid: str, fpath: str):
    base = (PROJECTS_DIR / pid).resolve()
    target = (base / fpath).resolve()
    if not str(target).startswith(str(base) + "/") and target != base:
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    if not target.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(target)


# --------------------------------------------------------------------------- #
# Incremental "growing map" sessions
# --------------------------------------------------------------------------- #
class IncrementalSession:
    def __init__(self, sid, stitcher):
        self.sid = sid
        self.stitcher = stitcher
        self.events = []
        self.lock = threading.Lock()
        self.finished = False
        self.reporter = ProgressReporter(emit=self.events.append,
                                         stages=INCR_STAGES)
        self.stitcher.reporter = self.reporter


def _save_map(sid, map_img):
    d = INCR_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "map.jpg"), map_img, [cv2.IMWRITE_JPEG_QUALITY, 95])


@app.post("/api/incremental/start")
def incremental_start(
    file: UploadFile = File(...),
    filename_template: str = Form(DEFAULT_TEMPLATE),
    engine: str = Form("superglue"),
    weights: str = Form("outdoor"),
    max_keypoints: int = Form(1024),
    match_threshold: float = Form(0.2),
    feature_max_dim: int = Form(1024),
    sinkhorn_iterations: int = Form(50),
    alignment: Literal["homography", "translation"] = Form("homography"),
    ransac_thresh: float = Form(3.0),
    refine: bool = Form(False),
    blend_levels: int = Form(6),
    exposure: bool = Form(True),
    min_extend_ratio: float = Form(0.0),
    min_extend_px: float = Form(0.0),
):
    try:
        position = require_positions([file.filename or ""], filename_template)[0]
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc), "events": []}, status_code=400)
    sid = uuid.uuid4().hex[:12]
    cfg = _stitch_config(engine, weights, max_keypoints, match_threshold,
                         feature_max_dim, sinkhorn_iterations, ransac_thresh,
                         "middle", refine, blend_levels, exposure, False,
                         min_extend_ratio=min_extend_ratio,
                         min_extend_px=min_extend_px, alignment=alignment)
    stitcher = IncrementalStitcher(
        cfg, download_url=f"/api/incremental/{sid}/map.jpg")
    session = IncrementalSession(sid, stitcher)
    session.filename_template = filename_template
    session.grid_positions = {position}
    incremental_sessions[sid] = session
    _prune_incremental()

    before = len(session.events)
    try:
        img = _read_upload(file)
        stitcher.seed(img)
        _save_map(sid, stitcher.map_img)
        return {"ok": True, "message": "Map created", "session_id": sid,
                "count": stitcher.count, "events": session.events[before:]}
    except StitchError as exc:
        return {"ok": False, "message": str(exc), "session_id": sid,
                "events": session.events[before:]}


@app.post("/api/incremental/add")
def incremental_add(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    session = incremental_sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": "Unknown session"}, status_code=404)
    if session.finished:
        return JSONResponse({"ok": False,
                             "message": "This session is finished."})
    try:
        position = require_positions([file.filename or ""], session.filename_template)[0]
        img = _read_upload(file)
    except (StitchError, ValueError) as exc:
        return {"ok": False, "message": str(exc), "events": []}

    before = len(session.events)
    with session.lock:
        try:
            if position in session.grid_positions:
                raise StitchError(f"Row {position[0]}, column {position[1]} is already in the map.")
            if not any(abs(position[0] - r) + abs(position[1] - c) == 1 for r, c in session.grid_positions):
                raise StitchError("Add an image in a neighboring row or column of the current grid first.")
            session.stitcher.add(img)
            session.grid_positions.add(position)
            ok, message = True, f"Added image {session.stitcher.count}"
            _save_map(session.sid, session.stitcher.map_img)
        except StitchError as exc:
            ok, message = False, str(exc)
    return {"ok": ok, "message": message,
            "count": session.stitcher.count,
            "events": session.events[before:]}


@app.post("/api/incremental/{sid}/finish")
def incremental_finish(sid: str):
    session = incremental_sessions.get(sid)
    if session is None:
        return JSONResponse({"error": "Unknown session"}, status_code=404)
    session.finished = True
    return {"ok": True}


@app.delete("/api/incremental/{sid}")
def incremental_delete(sid: str):
    incremental_sessions.pop(sid, None)
    shutil.rmtree(INCR_DIR / sid, ignore_errors=True)
    return {"deleted": sid}


@app.get("/api/incremental/{sid}/map.jpg")
def incremental_map(sid: str):
    p = INCR_DIR / sid / "map.jpg"
    if not p.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg")


def _prune_incremental(max_sessions=20):
    if len(incremental_sessions) <= max_sessions:
        return
    for sid in sorted(list(incremental_sessions))[: len(incremental_sessions) - max_sessions]:
        incremental_sessions.pop(sid, None)
        shutil.rmtree(INCR_DIR / sid, ignore_errors=True)


def _prune_jobs(max_jobs=40):
    if len(jobs) <= max_jobs:
        return
    for job_id in sorted(list(jobs))[: len(jobs) - max_jobs]:
        job = jobs.pop(job_id, None)
        if job and job.upload_dir:
            shutil.rmtree(job.upload_dir, ignore_errors=True)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    UPLOAD_DIR.mkdir(exist_ok=True)
    PROJECTS_DIR.mkdir(exist_ok=True)
    INCR_DIR.mkdir(exist_ok=True)
    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    if ":" in browser_host:
        browser_host = f"[{browser_host}]"
    base = f"http://{browser_host}:{args.port}"
    print(f"\n  Stitch Lab\n\n  Overview:             {base}/\n"
          f"  Pipeline comparison:  {base}/compare\n"
          f"  Microscope mosaic:    {base}/microscope\n"
          f"  Classic stitcher:     {base}/classic\n"
          f"  Grow a mosaic:        {base}/guided\n\n"
          "  Press Ctrl+C to stop the server.\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
