import cv2
import numpy as np
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# ---- CONFIGURATION ----
VIDEO_PATH = "RoadAccident.mp4"
KEYFRAME_DIR = "collision_keyframes"
CONFIDENCE_THRESHOLD = 0.4
COLLISION_SCORE_THRESHOLD = 5
MIN_OBJECTS_FOR_COLLISION = 1
DISPLAY_FRAMES = True
SAVE_FRAMES = True
CONTEXT_FRAME_WINDOW = 1  # Reduced for speed
FRAME_SKIP = 10 # Skip every 10 frames for faster processing

# ---- LOAD MODELS ----
print(" Loading models...")
yolo_models = [
    YOLO("yolov8n.pt"), 
    YOLO("runs/detect/train/weights/best.pt"),
    YOLO("runs/detect/train4/weights/best.pt"),
    YOLO("runs/detect/train2/weights/best.pt"),
    YOLO("runs/detect/train5/weights/best.pt"),
    YOLO("runs/detect/train7/weights/best.pt")
]

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# ---- UTILITY FUNCTIONS ----
def convert_frames_to_time(frame_number, fps):
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt")
    out = caption_model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_contextual_caption(cap, frame_index, fps):
    frames = []
    for offset in range(-CONTEXT_FRAME_WINDOW, CONTEXT_FRAME_WINDOW + 1):
        target_index = frame_index + offset
        if target_index < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    combined_caption = ""
    seen = set()
    for f in frames:
        caption = generate_caption(f)
        if caption not in seen:
            combined_caption += f" {caption.strip('.')}."
            seen.add(caption)
    return combined_caption.strip()

def detect_collision_frame(image):
    all_labels = []
    all_centers = []

    for model in yolo_models:
        results = model.predict(image, save=False, verbose=False)[0]
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            if label in ["car", "truck", "bus", "motorbike", "person"]:
                all_labels.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                all_centers.append((cx, cy))

    if len(all_labels) >= MIN_OBJECTS_FOR_COLLISION:
        for i in range(len(all_centers)):
            for j in range(i + 1, len(all_centers)):
                dist = np.linalg.norm(np.array(all_centers[i]) - np.array(all_centers[j]))
                if dist < 200:
                    return True, all_labels
        if len(all_labels) >= 3:
            return True, all_labels

    return False, all_labels

# ---- MAIN PIPELINE ----
def main():
    if SAVE_FRAMES and not os.path.exists(KEYFRAME_DIR):
        os.makedirs(KEYFRAME_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    prev_gray = None
    seen_timestamps = set()
    collision_data = []

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.sum(diff) / diff.size

        prev_gray = gray.copy()

        if motion_score > COLLISION_SCORE_THRESHOLD:
            is_collision, objects = detect_collision_frame(frame)
            if is_collision:
                timestamp = convert_frames_to_time(frame_count, fps)

                if timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    caption = generate_contextual_caption(cap, frame_count, fps)

                    if SAVE_FRAMES:
                        filename = os.path.join(KEYFRAME_DIR, f"collision_{frame_count}_{timestamp.replace(':', '-')}.jpg")
                        cv2.imwrite(filename, frame)

                    if DISPLAY_FRAMES:
                        cv2.imshow(f"Collision at {timestamp}", frame)
                        cv2.waitKey(500)
                        cv2.destroyAllWindows()

                    collision_data.append({
                        "timestamp": timestamp,
                        "objects": objects,
                        "caption": caption
                    })

        frame_count += FRAME_SKIP

    cap.release()

    # ---- FINAL SUMMARY ----
    print("\n📋 Detected Keyframes Summary:\n")
    if not collision_data:
        print("No major key frames detected.")
    else:
        for data in collision_data:
            print(f"⏳ Timestamp: {data['timestamp']}")
            print(f"   - Objects Detected: {', '.join(data['objects'])}")
            print(f"   - Scene Description: {data['caption']}\n")

if __name__ == "__main__":
    main()