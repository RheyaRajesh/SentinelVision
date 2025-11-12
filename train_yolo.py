from ultralytics import YOLO
import os

# Check if data.yaml exists
if not os.path.exists('data_yolo/data.yaml'):
    print("Error: Run preprocess.py first to generate data_yolo/")
    exit(1)

# Load pre-trained model
model = YOLO('yolov8n.pt')  # Nano for faster training

# Train
results = model.train(
    data='data_yolo/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='atm_surveillance',
    project='models/runs'  # Save here
)

# Export best model
best_model = 'models/runs/atm_surveillance/weights/best.pt'
os.rename(results.save_dir + '/weights/best.pt', best_model)
print(f"Trained model saved to {best_model}")