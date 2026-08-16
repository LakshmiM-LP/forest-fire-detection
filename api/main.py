from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
import cv2
import numpy as np


# ==========================================
# Create FastAPI application
# ==========================================

app = FastAPI(
    title="Forest Fire & Smoke Detection API",
    description="API for Fire and Smoke detection using YOLO",
    version="1.0.0"
)


# ==========================================
# Allow frontend to communicate with API
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Shared detection settings
# (kept identical to src/inference/video_temporal.py
# so image and video results don't silently disagree)
# ==========================================

MODEL_PATH = "models/fire_smoke_yolo11n_best.pt"
CONFIDENCE_THRESHOLD = 0.50
TARGET_CLASSES = ["fire", "smoke"]


# ==========================================
# Load trained YOLO model
# ==========================================

model = YOLO(MODEL_PATH)
print("Fire & Smoke YOLO model loaded successfully!")


# ==========================================
# Health check
# ==========================================

@app.get("/")
def home():
    return {"message": "Forest Fire & Smoke Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}


# ==========================================
# Image prediction endpoint
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "success": False,
            "message": "Invalid image file"
        }

    results = model.predict(
        source=image,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
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

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3)
            })

    detected = len(detections) > 0

    return {
        "success": True,
        "detected": detected,
        "detections": detections
    }