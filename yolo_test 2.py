import cv2
import numpy as np
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json

# Load YOLOv8 model for object detection
yolo_model = YOLO("/Users/purneshbr/Desktop/major_project/runs/detect/train4/weights/best.pt")
if not hasattr(yolo_model, 'names') or not yolo_model.names:
    raise Exception("❌ YOLO model did not load correctly or labels missing.")

# Load BLIP caption model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.4

def convert_frames_to_time(frame_number, fps):
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def extract_keyframes(video_path, threshold=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video file.")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes, keyframe_timestamps = [], []
    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(gray_frame, prev_frame)
            score = np.mean(diff)
            if score > threshold:
                timestamp = convert_frames_to_time(frame_count, fps)
                keyframes.append(frame)
                keyframe_timestamps.append(timestamp)
                print(f"✅ Keyframe detected at {timestamp}")

        prev_frame = gray_frame
        frame_count += 1

    cap.release()
    return keyframes, keyframe_timestamps

def detect_objects(image):
    results = yolo_model(image)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = f"{yolo_model.names[class_id]}: {conf:.2f}"
            detected_objects.append(yolo_model.names[class_id])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, detected_objects

def generate_caption(image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_image, return_tensors="pt")
        out = caption_model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️ BLIP captioning failed: {e}")
        return "Caption unavailable"

# Run the full pipeline
video_path = "sample.mp4"
keyframes, timestamps = extract_keyframes(video_path)

keyframe_data = []

for idx, (frame, timestamp) in enumerate(zip(keyframes, timestamps)):
    print(f"\n🔍 Processing keyframe at {timestamp}...")
    detected_frame, objects = detect_objects(frame)
    caption = generate_caption(frame)

    keyframe_data.append({
        "timestamp": timestamp,
        "objects": objects,
        "caption": caption
    })

    # Display the detected keyframe
    cv2.imshow(f"Keyframe at {timestamp}", detected_frame)
    print("📸 Objects:", objects)
    print("📝 Caption:", caption)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

# Save results to JSON
with open("keyframe_results.json", "w") as f:
    json.dump(keyframe_data, f, indent=2)

# Final summary printout
print("\n📌 FINAL KEYFRAME ANALYSIS REPORT:\n")
for data in keyframe_data:
    print(f"⏳ {data['timestamp']}")
    print(f"   - Objects Detected: {', '.join(data['objects']) if data['objects'] else 'None'}")
    print(f"   - Scene Description: {data['caption']}\n")