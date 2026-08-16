"""
Reproducible Training Pipeline for Smart Campus Waste AI Computer-Vision Model.
Fine-tunes Ultralytics YOLO (YOLO26m / YOLOv8m) on custom waste dataset.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ERROR] Ultralytics package is not installed. Please run: pip install ultralytics")

def run_training(
    data_yaml: str = "waste_dataset/data.yaml",
    model_name: str = "yolov8m.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    project: str = "yolo_service/runs",
    name: str = "waste_model_v1"
):
    """
    Executes Ultralytics YOLO fine-tuning training on custom waste dataset.
    Saves best.pt, last.pt, confusion matrix, and training performance logs.
    """
    if not YOLO_AVAILABLE:
        print("[ABORT] Cannot execute training without ultralytics installed.")
        return None

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        print(f"[ERROR] Specified data.yaml not found at: {yaml_path.resolve()}")
        print("[NOTICE] Please place your annotated waste images in waste_dataset/ and verify data.yaml.")
        return None

    print("=" * 60)
    print("STARTING CUSTOM WASTE AI MODEL FINE-TUNING")
    print(f"Base Checkpoint: {model_name}")
    print(f"Dataset Config:  {data_yaml}")
    print(f"Epochs:          {epochs} (Early Stopping Patience: {patience})")
    print(f"Image Resolution: {imgsz}x{imgsz}")
    print("=" * 60)

    try:
        model = YOLO(model_name)
        
        # Train on custom waste dataset
        results = model.train(
            data=str(yaml_path.resolve()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=project,
            name=name,
            exist_ok=True,
            save=True,
            plots=True,
            workers=4
        )

        weights_dir = Path(project) / name / "weights"
        best_weights = weights_dir / "best.pt"
        if best_weights.exists():
            print(f"[SUCCESS] Custom Waste Model successfully trained and saved at: {best_weights.resolve()}")
            print(f"[ACTION] Copy {best_weights.name} to yolo_service/weights/best.pt for production inference.")
        return results

    except Exception as err:
        print(f"[TRAINING ERROR] Failed to execute training pipeline: {err}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom Waste YOLO Model")
    parser.add_argument("--data", type=str, default="waste_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model weights (e.g. yolo26m.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    args = parser.parse_args()
    run_training(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience
    )
