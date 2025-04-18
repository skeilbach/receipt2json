# receipt2json

Receipt extraction library using classical computer vision methods as well as leveraging deep-learning approaches.

---

## Background

This project aims to introduce a pipeline to extract structured data from supermarket receipts, converting them into a clean, machine-readable JSON format. It's built with PyTorch and Ultralytics YOLO for detection and OCR, with a focus on German grocery store receipts - but can theoretically extended to receipts of any language given its modular, expandable nature.

## Features

 - 📷 Robust image preprocessing: Adequate preprocessing can drastically enhance the ocr results which is why extra attention was placed on creating a robust preprocessing pipeline.
 - 🧾 Line-item and price extraction to JSON
 - ⚙️ Carefully curated training-dataset for receipt instance segmentation: Annotated 353 images (taken from [Receipt ORC](https://universe.roboflow.com/cj-gl8m1/receipt-ocr-cvfxx)) to train a custom instance segmentation model on and uploaded it to [Roboflow](https://universe.roboflow.com/receiptsegmentation/receipt-instance-segmentation)

---

## Installation

Ensure that **Python 3.10.12** and **uv** is installed. Then clone the repository and install packages

```bash
git clone https://github.com/yourusername/receipt2json.git
cd receipt2json
python -m venv .venv
source .venv/bin/activate
uv pip install -e .
```

To use the pre-commit tool also do:
```bash
uv pip install .[format]
pre-commit install
```

## Usage
A training script to train a custom instance segmentation model can be found in `receipt_scanner_train.py`. Its current features are:
```bash
python receipt scanner_train.py [--model-path]
                                [--dataset-path]
                                [--epochs]

optional arguments:
  --model-path     Path to the model weights
  --dataset-path   Path to the dataset used for fine-tuning
  --epochs         Number of training epochs
```


## ToDo
- [] Add page dewarping (see [page-dewarp](https://github.com/mzucker/page_dewarp/tree/master) and [ACTM](https://ieeexplore.ieee.org/document/8892891)
- [] Train OCR model
  - [] Collect training data
  - [] Choose model
