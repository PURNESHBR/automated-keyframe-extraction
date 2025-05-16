import os
import pandas as pd

# Base path to your runs directory
base_path = 'runs/detect'

# Initialize best values
best_map = 0
best_model_path = None

# Loop through all train folders
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    results_csv = os.path.join(folder_path, 'results.csv')
    weights_path = os.path.join(folder_path, 'weights', 'best.pt')

    # Check if results.csv and best.pt exist
    if os.path.exists(results_csv) and os.path.exists(weights_path):
        try:
            df = pd.read_csv(results_csv)
            max_map = df['metrics/mAP_0.5'].max()  # or 'map' if you want mAP@0.5:0.95

            if max_map > best_map:
                best_map = max_map
                best_model_path = weights_path
        except Exception as e:
            print(f"Skipping {folder_path} due to error: {e}")

if best_model_path:
    print(f"\n✅ Best model: {best_model_path} with mAP@0.5 = {best_map:.4f}")
    
    # Load and run the best model
    from ultralytics import YOLO
    model = YOLO(best_model_path)
    # Example: run on a sample video/image
    results = model('/Users/purneshbr/Desktop/major_project/sample.mp4', save=True)
else:
    print("❌ No valid models found.")