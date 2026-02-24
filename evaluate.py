from ultralytics import YOLO
import torch

def evaluate_model():
    # 1. Check if GPU is available
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load best trained model
    model = YOLO("detect_yolov8s/traffic_model_v8s5/weights/best.pt")

    metrics = model.val(
        data="traffic.yaml",
        split="test",
        imgsz=640,
        device=device
    )

    print("\n===== TEST RESULTS =====")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
