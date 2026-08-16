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
# Load trained YOLO model
# ==========================================

MODEL_PATH = "models/fire_smoke_yolo11n_best.pt"

model = YOLO(MODEL_PATH)

print("Fire & Smoke YOLO model loaded successfully!")


# ==========================================
# Health check
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Forest Fire & Smoke Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded"
    }


# ==========================================
# Image prediction endpoint
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded file
    contents = await file.read()

    # Convert bytes to numpy array
    image_array = np.frombuffer(contents, np.uint8)

    # Convert numpy array to OpenCV image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "success": False,
            "message": "Invalid image file"
        }

    # Run YOLO detection
    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3)
            })

    # Check whether fire/smoke detected
    detected = len(detections) > 0

    return {
        "success": True,
        "detected": detected,
        "detections": detections
    }