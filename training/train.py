#!/usr/bin/env python3
"""Train YOLO26s on CAW with MLflow logging.

Requires: pip install ultralytics mlflow
Start the MLflow server before running this script.
"""
import os
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent


def main():
    weights = TRAINING_DIR / "model_weights/yolo26s.pt"
    data = PROJECT_DIR / "data/YOLO_DATA/CAW/data.yaml"
    if not weights.is_file():
        raise SystemExit("Missing weights. Run: python3 training/download_yolo26s.py (from the project root)")
    if not data.is_file():
        raise SystemExit(f"Dataset configuration not found: {data}")

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "crop-weed-yolo26")
    os.environ.setdefault("MLFLOW_RUN", "yolo26s-caw-full-baseline")

    from ultralytics import YOLO, settings

    settings.update({"mlflow": True})
    model = YOLO(str(weights))
    model.train(
        data=str(data),
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        seed=42,
        project=str(TRAINING_DIR / "runs"),
        name=os.environ["MLFLOW_RUN"],
    )


if __name__ == "__main__":
    main()
