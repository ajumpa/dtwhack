# Crop and weed detection

YOLO26s object detection with two classes: `crop` and `weed`. The datasets live outside Git; training runs and MLflow history are kept in `training/`.

## Layout

- `scripts/convert_caw.py`, `scripts/convert_acre.py`: convert source annotations to YOLO boxes.
- `scripts/view_yolo.py`: browse images, labels, and model predictions.
- `training/download_yolo26s.py`: download pretrained YOLO26s weights.
- `training/train.py`: train the CAW baseline.
- `training/train_subset.py`: fine-tune the baseline on the mixed CAW/ACRE training list.
- `training/runs/`: checkpoints, metrics, and plots from completed runs.
- `training/mlflow.db`, `training/mlartifacts/`: local MLflow run history.
- `data/`: local datasets; ignored by Git. CAW and ACRE images and labels are **not** included in this repository.

## Set up

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python training/download_yolo26s.py
```

CAW and ACRE data must be obtained separately if you want to rerun the recorded experiments. Put the converted datasets at `data/YOLO_DATA/CAW/` and `data/YOLO_DATA/ACRE/`. The mixed training list in CAW also references ACRE training images. Update `path:` in `CAW/data.yaml` to its absolute directory on your machine before running the baseline script; the fine-tuning script generates its own config.

For an NVIDIA GPU, install the PyTorch build appropriate for your CUDA setup before `requirements.txt`; otherwise PyTorch is installed as a dependency of Ultralytics.

## Add new data

Place a new YOLO detection dataset under `data/YOLO_DATA/<dataset-name>/`:

```text
<dataset-name>/
  images/                 # JPG or PNG images
  labels/                 # one .txt file per labeled image, same base name
  train.txt               # training image paths
  val.txt                 # validation image paths
  test.txt                # test image paths
  data.yaml
```

Each label line is `class_id center_x center_y width height`, with coordinates normalized to the image size (0–1). Use `0` for crop and `1` for weed. For example, `0 0.500 0.500 0.200 0.300` labels one crop. Each split file contains one image path per line, such as `./images/example.jpg`. Keep the splits disjoint.

`data.yaml` should contain:

```yaml
path: /absolute/path/to/data/YOLO_DATA/<dataset-name>
train: train.txt
val: val.txt
test: test.txt
names:
  0: crop
  1: weed
```

The viewer discovers new datasets automatically. To train on a new dataset, pass its `data.yaml` to Ultralytics or update the dataset path in the training script.

## Train with MLflow

Start the tracking server from the project root:

```bash
cd training
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts
```

Open <http://127.0.0.1:5000>. In another terminal, activate `.venv` and run from the project root:

```bash
python training/train.py          # baseline: all CAW training images
python training/train_subset.py   # mixed CAW/ACRE fine-tuning from baseline best.pt
```

The fine-tuning script writes `training/data_subset.yaml` using the current dataset path. Both scripts use CAW validation data; the test split is reserved for final evaluation. Run outputs go to `training/runs/`.

## View data and predictions

```bash
python scripts/view_yolo.py
```

Open <http://127.0.0.1:8000>. Select a dataset and split, then use **Run model** to overlay predictions from a saved checkpoint on the ground-truth boxes. The viewer defaults to CAW validation images.
