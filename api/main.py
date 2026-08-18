import asyncio
import base64
import subprocess
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO

from src.inference.temporal_detector import TemporalFireDetector


# ==========================================
# App
# ==========================================

app = FastAPI(
    title="Forest Fire & Smoke Detection API",
    description="API for Fire and Smoke detection using YOLO",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Shared settings
# (kept identical to src/inference/video_temporal.py so image,
# video, and live results don't silently disagree)
# ==========================================

MODEL_PATH = "models/fire_smoke_yolo11n_best.pt"
CONFIDENCE_THRESHOLD = 0.50
TARGET_CLASSES = ["fire", "smoke"]

VIDEO_UPLOAD_DIR = Path("data/videos")
RUNS_DIR = Path("runs")
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Load model once, shared by all endpoints
# ==========================================

model = YOLO(MODEL_PATH)
print("Fire & Smoke YOLO model loaded successfully!")


# ==========================================
# Health
# ==========================================

@app.get("/")
def home():
    return {"message": "Forest Fire & Smoke Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}


# ==========================================
# Image prediction
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "message": "Invalid image file"}

    results = model.predict(
        source=image, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            if class_name.lower() not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "box": [x1, y1, x2, y2],
            })

    return {
        "success": True,
        "detected": len(detections) > 0,
        "detections": detections,
    }


# ==========================================
# Video upload -> background job
#
# Reuses src/inference/video_temporal.py as a subprocess. That
# script is already tested end-to-end (temporal debounce, audio
# alerts, H264 export); wrapping it rather than reimplementing it
# here avoids re-introducing bugs we already fixed once.
# ==========================================

video_jobs = {}  # job_id -> {"status", "process", "input_path", "output_path", "error"}


def _run_video_job(job_id: str, input_path: Path):

    output_path = RUNS_DIR / "temporal_detection_h264.mp4"
    if output_path.exists():
        output_path.unlink()

    process = subprocess.Popen(
        ["python", "src/inference/video_temporal.py", str(input_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    video_jobs[job_id]["process"] = process
    video_jobs[job_id]["status"] = "processing"

    log_lines = []
    for line in process.stdout:
        log_lines.append(line.rstrip())
        video_jobs[job_id]["log"] = log_lines[-30:]  # keep it bounded

    process.wait()

    if process.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1024:
        video_jobs[job_id]["status"] = "done"
        video_jobs[job_id]["output_path"] = str(output_path)
    else:
        video_jobs[job_id]["status"] = "error"
        video_jobs[job_id]["error"] = "\n".join(log_lines[-15:])


@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):

    job_id = str(uuid.uuid4())
    input_path = VIDEO_UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    video_jobs[job_id] = {
        "status": "queued",
        "process": None,
        "input_path": str(input_path),
        "output_path": None,
        "error": None,
        "log": [],
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_video_job, job_id, input_path)

    return {"job_id": job_id, "status": "queued"}


@app.get("/video/status/{job_id}")
async def video_status(job_id: str):

    job = video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    return {
        "job_id": job_id,
        "status": job["status"],
        "log": job.get("log", []),
        "error": job.get("error"),
    }


@app.get("/video/result/{job_id}")
async def video_result(job_id: str):

    job = video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not finished (status: {job['status']})")

    return FileResponse(
        job["output_path"],
        media_type="video/mp4",
        filename="fire_smoke_detection_result.mp4",
    )


# ==========================================
# Live camera -- WebSocket
#
# Browser captures webcam frames to a canvas, sends each as a JPEG
# (base64) over the socket, server runs one YOLO pass + the shared
# temporal state machine, and returns detections + alert status.
# One TemporalFireDetector instance per connection, so concurrent
# viewers don't share event state.
# ==========================================

@app.websocket("/ws/live")
async def live_detection(websocket: WebSocket):

    await websocket.accept()

    detector = TemporalFireDetector(
        model=model,
        target_classes=TARGET_CLASSES,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        assumed_fps=4.0,  # browser sends a few frames/sec, not full video fps
    )

    start_time = time.time()

    try:
        while True:
            message = await websocket.receive_text()

            if "," in message:
                message = message.split(",", 1)[1]  # strip data URL prefix if present

            try:
                frame_bytes = base64.b64decode(message)
                frame_array = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            height, width = frame.shape[:2]
            current_time = time.time() - start_time

            detections, status = detector.process_frame(
                frame, current_time, width, height
            )

            await websocket.send_json({
                "detections": detections,
                "status": status,
            })

    except WebSocketDisconnect:
        pass