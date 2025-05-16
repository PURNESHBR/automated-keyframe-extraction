import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

# === CONFIGURATION ===
video_path = '/Users/purneshbr/Desktop/major_project/sample.mp4'
model_paths = [
    'runs/detect/train2/weights/best.pt',
    'runs/detect/train4/weights/best.pt'
]
yaml_path = '/Users/purneshbr/Desktop/major_project/Accident Detection/data.yaml'
output_keyframe_dir = 'keyframes'
os.makedirs(output_keyframe_dir, exist_ok=True)

confidence_threshold = 0.5
ssim_threshold = 0.90  # similarity threshold (lower = more difference required)

# === Load class names ===
with open(yaml_path, 'r') as yml:
    names = yaml.safe_load(yml)['names']

# === Load Models ===
models = [YOLO(path) for path in model_paths]

# === Setup video capture ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_id = 0
summary = []
last_keyframe = None

def get_combined_detections(detection_results, conf_thresh=0.5):
    all_boxes = []
    for result in detection_results:
        boxes = result.boxes
        for i in range(len(boxes)):
            if boxes.conf[i] >= conf_thresh:
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                all_boxes.append((xyxy, conf, cls))
    return all_boxes

def is_significantly_different(frame1, frame2, threshold=ssim_threshold):
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score < threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run ensemble predictions
    detections = []
    for model in models:
        results = model.predict(source=frame, conf=confidence_threshold, verbose=False)[0]
        detections.append(results)

    combined_detections = get_combined_detections(detections)

    if combined_detections:
        if last_keyframe is None or is_significantly_different(frame, last_keyframe):
            # Save keyframe
            timestamp = frame_id / fps
            filename = f"{output_keyframe_dir}/frame_{frame_id}_t{int(timestamp)}s.jpg"
            cv2.imwrite(filename, frame)
            last_keyframe = frame.copy()

            # Add to summary
            detected_names = [names[cls] for (_, _, cls) in combined_detections]
            summary.append(f"Frame {frame_id} (Time {timestamp:.2f}s): Detected {len(detected_names)} objects - {detected_names}")

    frame_id += 1

cap.release()

# === Save Summary ===
with open("summary.txt", "w") as f:
    f.write("YOLO Ensemble Keyframe Summary:\n")
    f.write("\n".join(summary))

print("✅ Keyframe detection complete.")
print("📄 Summary saved to summary.txt")