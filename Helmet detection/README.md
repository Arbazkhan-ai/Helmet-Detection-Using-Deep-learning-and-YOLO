<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Helmet Detection Using YOLO</title>
</head>
<body>
    <h1>🪖 Helmet Detection Using YOLO</h1>
    <p><strong>Real-time Helmet Detection</strong> system using Deep Learning and YOLO. Detects helmets in images, videos, and live camera feeds to improve workplace safety.</p>

    <h2>Features</h2>
    <ul>
        <li>Detect helmets in real-time from webcam or video</li>
        <li>Supports training custom dataset</li>
        <li>Fast and accurate object detection using YOLOv8</li>
        <li>Displays bounding boxes and confidence scores</li>
    </ul>

    <h2>Installation</h2>
    <pre>
# Clone the repo
git clone https://github.com/Arbazkhan-ai/Helmet-Detection-Using-Deep-learning-and-YOLO.git

# Move into project folder
cd Helmet-Detection-Using-Deep-learning-and-YOLO

# Install dependencies
pip install ultralytics opencv-python
    </pre>

    <h2>Usage</h2>
    <pre>
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
    </pre>

    <h2>Dataset Structure</h2>
    <pre>
dataset/
├─ train/
├─ valid/
├─ test/
└─ data.yaml
    </pre>

    <h2>Demo</h2>
    <p>Replace <code>demo.jpg</code> with your demo image path if needed.</p>

    <h2>Contributing</h2>
    <p>Contributions are welcome! Open an issue or submit a pull request.</p>

    <h2>License</h2>
    <p>MIT License</p>

    <p><a href="https://github.com/Arbazkhan-ai/Helmet-Detection-Using-Deep-learning-and-YOLO">View on GitHub</a></p>
</body>
</html>
