from ultralytics import YOLO

# Load trained fire and smoke model
model = YOLO("models/fire_smoke_yolo11n_best.pt")

# Test image
image_path = "data/dataset/test/images/002467_jpg.rf.d303a7b3afbebd05c890e3e3c515af95.jpg"

# Run detection
results = model.predict(
    source=image_path,
    imgsz=640,
    conf=0.25
)

# Save detection results
for result in results:
    result.save(filename="runs/detection_result.jpg")

print("Detection completed!")
print("Result saved to: runs/detection_result.jpg")