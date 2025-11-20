<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Helmet Detection Using YOLO</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }
        h1, h2 {
            text-align: center;
            color: #0b3d91;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background-color: #fff;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        code {
            background-color: #eee;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 0.95em;
        }
        pre {
            background-color: #eee;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        img {
            max-width: 100%;
            border-radius: 10px;
            display: block;
            margin: 10px auto;
        }
        a.button {
            display: inline-block;
            margin: 10px 0;
            padding: 10px 20px;
            background-color: #0b3d91;
            color: #fff;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }
        a.button:hover {
            background-color: #0650a4;
        }
        ul {
            margin-left: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🪖 Helmet Detection Using YOLO</h1>
        <p><strong>Real-time Helmet Detection</strong> system using Deep Learning and YOLO. Detects helmets in images, videos, and live camera feeds to improve workplace safety and automated monitoring.</p>

        <h2>📦 Features</h2>
        <ul>
            <li>Detect helmets in real-time from webcam or video</li>
            <li>Supports training custom dataset</li>
            <li>Fast and accurate object detection using YOLOv8</li>
            <li>Displays bounding boxes and confidence scores</li>
        </ul>

        <h2>🛠 Installation</h2>
        <pre>
# Clone the repo
git clone https://github.com/Arbazkhan-ai/Helmet-Detection-Using-Deep-learning-and-YOLO.git

# Move into project folder
cd Helmet-Detection-Using-Deep-learning-and-YOLO

# Install dependencies
pip install ultralytics opencv-python
        </pre>

        <h2>🚀 Usage</h2>
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

        <h2>📁 Dataset Structure</h2>
        <pre>
dataset/
├─ train/
├─ valid/
├─ test/
└─ data.yaml
        </pre>

        <h2>📷 Demo</h2>
        <img src="images/demo.png" alt="Helmet Detection Demo">

        <h2>💡 Contributing</h2>
        <p>Contributions are welcome! Please open an issue or submit a pull request for improvements.</p>

        <h2>⚡ License</h2>
        <p>This project is licensed under the MIT License.</p>

        <p style="text-align:center;"><a href="https://github.com/Arbazkhan-ai/Helmet-Detection-Using-Deep-learning-and-YOLO" class="button">View on GitHub</a></p>
    </div>
</body>
</html>
