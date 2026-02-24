from ultralytics import YOLO
import torch

def train_traffic_model():
    # 1. Check if GPU is available (Training on CPU is 20x slower)
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Load the 'Nano' model (yolov11n)
    # This is the best choice for real-time counting because it's fast.
    model = YOLO('yolov8s.pt') 

    # 3. Start training
    model.train(
        data='traffic.yaml',    # Path to your yaml file
        epochs=100,           # Start with 50; you can stop early if it's accurate
        imgsz=640,           # Standard size for traffic detection
        batch=16,            # Reduce to 8 if you get 'Out of Memory' errors
        device=device,       # Uses your GPU
        patience=10,        # Stop training if no improvement for 10 epochs
        cos_lr=True,          # Use cosine learning rate schedule for better convergence
        lr0=0.01,             # Initial learning rate
        name='traffic_model_v8s', # Folder name for 
        
    )

if __name__ == '__main__':
    train_traffic_model()