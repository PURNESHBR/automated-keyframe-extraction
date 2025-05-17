import cv2
import numpy as np
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load multiple YOLO models for object detection 
yolo_models = [
    
    YOLO("runs/detect/train2/weights/best.pt"),
    YOLO("runs/detect/train4/weights/best.pt"),
    YOLO("runs/detect/train5/weights/best.pt")

]

# Load BLIP model for caption generation 

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def convert_frames_to_time(frame_number, fps):
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def extract_keyframes(video_path, threshold=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []
    keyframe_timestamps = []
    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(gray_frame, prev_frame)
            score = np.sum(diff) / diff.size

            if score > threshold:
                timestamp = convert_frames_to_time(frame_count, fps)
                keyframes.append(frame)
                keyframe_timestamps.append(timestamp)
                print(f"🔹 Keyframe detected at {timestamp}")

        prev_frame = gray_frame
        frame_count += 1

    cap.release()
    return keyframes, keyframe_timestamps

def detect_objects(image):
    detected_objects = []

    for model in yolo_models:
        results = model.predict(source=image, save=False, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = f"{model.names[class_id]}: {confidence:.2f}"
                detected_objects.append(model.names[class_id])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, detected_objects

def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Run the complete pipeline
video_path = "sample.mp4"
keyframes, timestamps = extract_keyframes(video_path)

keyframe_data = []

for idx, (frame, timestamp) in enumerate(zip(keyframes, timestamps)):
    detected_frame, objects = detect_objects(frame)
    caption = generate_caption(frame)

    keyframe_data.append({
        "timestamp": timestamp,
        "objects": objects,
        "caption": caption
    })

    # Display each keyframe briefly
    cv2.imshow(f"Keyframe at {timestamp}", detected_frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

# Final report
print("\n📌 **Keyframe Analysis Report:**\n")
for data in keyframe_data:
    print(f"⏳ Keyframe at {data['timestamp']}:")
    print(f"   - Objects Detected: {', '.join(data['objects']) if data['objects'] else 'None'}")
    print(f"   - Scene Description: {data['caption']}\n")