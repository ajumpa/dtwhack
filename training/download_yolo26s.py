#!/usr/bin/env python3
"""Download YOLO26s weights. Requires: pip install huggingface_hub"""
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    path = hf_hub_download(
        repo_id="Ultralytics/YOLO26",
        filename="yolo26s.pt",
        local_dir=Path(__file__).resolve().parent / "model_weights",
    )
    print(f"Weights saved to: {path}")


if __name__ == "__main__":
    main()
