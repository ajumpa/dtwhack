#!/usr/bin/env python3
"""Fine-tune the baseline's best weights on the mixed CAW and ACRE training list."""
import os
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO, settings

TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR.parent / "data/YOLO_DATA/CAW"


def main():
    run_name = "yolo26s-caw-acre-mixed-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "crop-weed-yolo26")
    os.environ["MLFLOW_RUN"] = run_name
    settings.update({"mlflow": True})

    model = YOLO(str(TRAINING_DIR / "runs/yolo26s-caw-full-baseline/weights/best.pt"))

    # Change only the training list; preserve the original validation/test splits.
    data = yaml.safe_load((DATASET_DIR / "data.yaml").read_text())
    data["path"] = str(DATASET_DIR)
    data["train"] = "train_subset.txt"
    subset_yaml = TRAINING_DIR / "data_subset.yaml"
    subset_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    model.train(
        data=str(subset_yaml),
        resume=False,
        epochs=50,
        patience=10,
        optimizer="AdamW",
        lr0=0.0001,
        warmup_bias_lr=0.0,
        imgsz=640,
        batch=8,
        mosaic=0.0,
        scale=0.2,
        freeze=None,
        seed=42,
        project=str(TRAINING_DIR / "runs"),
        name=run_name,
    )


if __name__ == "__main__":
    main()
