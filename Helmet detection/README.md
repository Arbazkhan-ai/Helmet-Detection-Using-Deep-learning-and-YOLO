# 🪖 Helmet Detection Using YOLO

Real-time Helmet Detection system using Deep Learning and YOLO. Detects helmets in images, videos, and live camera feeds to improve workplace safety.

## Features
- Detect helmets in real-time from webcam or video
- Supports training custom dataset
- Fast and accurate object detection using YOLOv8
- Displays bounding boxes and confidence scores

## Dataset Structure
dataset/
├─ train/
├─ valid/
├─ test/
└─ data.yaml


## Installation
```bash
# Clone the repo
git clone https://github.com/Arbazkhan-ai/Helmet-Detection-Using-Deep-learning-and-YOLO.git

# Move into project folder
cd Helmet-Detection-Using-Deep-learning-and-YOLO

# Install dependencies
pip install ultralytics opencv-python

# Train the model
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="dataset/data.yaml", epochs=50, imgsz=640)

# Run live helmet detection
import cv2
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Helmet Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

