import urllib.request
import os

url = "https://huggingface.co/deepghs/yolo-face/resolve/42342bcd13d9cc36c04d8b424b682d737dfd7082/yolov8n-face/model.onnx"
os.makedirs("models", exist_ok=True)
out_path = "models/yolov8n-face.onnx"

if not os.path.exists(out_path):
    print("Downloading YOLOv8n-face ONNX model...")
    urllib.request.urlretrieve(url, out_path)
    print("Downloaded successfully.")
else:
    print("Model already exists.")
