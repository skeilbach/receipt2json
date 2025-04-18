"""Custom training script specifically for fine-tuning yolo-based segmentation models."""

# Load packages
import argparse
import gc
import os

import torch
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO


def main(args):
    """Train YOLO model on custom dataset."""
    # ---------------------
    # Step 1: Load dataset
    # ---------------------
    # Define home variable to make managing datasets, images and models easier
    HOME = os.getcwd()

    if not args.dataset_path:
        # Load dataset from roboflow for training
        os.makedirs(f"{HOME}/datasets", exist_ok=True)
        os.chdir(f"{HOME}/datasets")

        load_dotenv()
        rf_api_key = os.getenv("ROBOFLOW_API_KEY")

        rf = Roboflow(api_key=rf_api_key)
        project = rf.workspace("receiptsegmentation").project(
            "receipt-instance-segmentation"
        )
        version = project.version(2)
        dataset = version.download("yolov11")

        os.chdir(f"{HOME}")

    # -------------------------------------------
    # Step 2: Fine-tune YOLO11 on custom dataset
    # -------------------------------------------

    # Load and train pre-trained yolo model
    dataset_path = dataset.location if not args.dataset_path else args.dataset_path
    print(f"[INFO] Loading model from {os.path.basename(args.model_path)}\n")
    print(f"[INFO] Starting training using dataset at location {dataset_path}\n")
    model = YOLO(args.model_path)

    if os.path.exists(f"{dataset_path}/data.yaml"):
        results = model.train(
            data=f"{dataset_path}/data.yaml", epochs=args.epochs, imgsz=640, plots=True
        )
    else:
        print(
            f"[ERROR] File {dataset_path}/data.yaml does not exist or is named differently. Please rename it and ensure the custom dataset is in YOLOv11 format!"
        )

    # Delete model and cache to free up space on GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Validate and test fine-tuned model
    model_ft = YOLO(f"{HOME}/runs/segment/train/weights/best.pt")  # Fine-tuned model
    validation_results = model_ft.val(data=f"{dataset_path}/data.yaml")

    model_ft.predict(f"{dataset_path}/test/images", save=True, imgsz=640, conf=0.5)

    return results, validation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model weights",
        default="yolo11s-seg.pt",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset used for fine-tuning",
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=35)

    args = parser.parse_args()

    main(args)
