from __future__ import division 
from sklearn.cluster import KMeans
import numpy as np
import yaml
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch
# Define helper functions for the main classes


###
# Custom Receipt Scanner
###

def extend_segmentation_mask(mask, scaling=1.1):
    """
    Extends the original segmentation mask into a larger background defined
    by the scaled height and width of the original mask to make edge detection
    easier in the case the segmentation mask extends beyond the borders of the image.

    Args:
        - mask (np.ndarray): The segmentation mask of the receipt from YOLO
        - scaling (float): Factor to scale the height and width of the mask with
    """
    height, width = mask.shape
    new_height, new_width = int(scaling*height), int(scaling*width)
    
    # Create a new black background 
    extended_mask = np.zeros((new_height, new_width), dtype=np.uint8)

    # Get the top-left corner where the original mask will be placed
    top_left_x = (new_width - mask.shape[1]) // 2
    top_left_y = (new_height - mask.shape[0]) // 2

    # Place the original mask onto the extended background
    extended_mask[top_left_y:top_left_y + mask.shape[0], top_left_x:top_left_x + mask.shape[1]] = mask

    return extended_mask

def transform_corners(corner_coordinates, initial_img_shape, final_img_shape):
    """
    Transforms corner coordinates of form (x,y) of a rectangle from an initial image size to fit the same
    proportions in the final image, keeping the aspect ratio. Essentially, resizing the original rectangle.

    Args:
     - initial_img_shape (tuple): The initial image dimensions
     - final_img_shape (tuple): The dimensions of the final image (height, width).

    Returns:
     - crns_reconstructed (np.ndarray): Resized corner coordinates of shape (4,2)
    """
    # Get the dimensions of the original image
    final_height, final_width = final_img_shape

    # Find the top-left corner of the original mask in the extended mask
    initial_height, initial_width = initial_img_shape
    top_left_x = (initial_width - final_width) // 2
    top_left_y = (initial_height - final_height) // 2

    # Reconstruct x,y coordinates for corner points 
    crns_reconstructed = []
    crns = corner_coordinates.reshape(4,2)
    for crn in crns:
        x,y = crn
        x -= top_left_x
        y -= top_left_y
        crns_reconstructed.append([x,y])

    return np.array(crns_reconstructed)

def load_yaml_config(config_path):
    """
    Load yaml config file that stores all important variables for the Custom Receipt Scanner methods
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def get_corners_cluster(rect, corners):
    """
    Compute clusters representing the top-left, top-right, bottom-left and bottom-right candidates for a polygon.

    Args: 
        - rect (np.ndarray): The inital centroids for our kmeans algorithm
        - corners (np.ndarray): The corner candidates

    Returns:
        - clusters (list[np.ndarray)]): A list of np arrays containing the x,y coordinates of the points which where assigned to each corner cluster
    """
    kmeans = KMeans(n_clusters=4, init=rect).fit(corners)

    # Get the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Get cluster_centers representing top-left, top-right, bottom-left and bottom-right cluster
    cluster_cent_rect = contour_to_rect(cluster_centers)
    (ctl, ctr, cbr, cbl) = (cluster_cent_rect) # Convert cluster_centers to float32 to match float32 precision of ctl/ctr/cbl/cbr
    
    # Group points by their label and cluster_center (the ordering of the cluster centers and lables is the same,
    # i.e. kmeans.cluster_centers_[0] = centroid of cluster 0)
    # Also, Convert cluster_centers to float32 to match float32 precision of ctl/ctr/cbl/cbr
    cluster_centers = np.float32(cluster_centers)
    cluster_tl = corners[labels == int(np.where(np.all(cluster_centers == ctl, axis=1))[0][0])]
    cluster_tr = corners[labels == int(np.where(np.all(cluster_centers == ctr, axis=1))[0][0])]
    cluster_br = corners[labels == int(np.where(np.all(cluster_centers == cbr, axis=1))[0][0])]
    cluster_bl = corners[labels == int(np.where(np.all(cluster_centers == cbl, axis=1))[0][0])]
    
    # Group clusters together to compute combinations
    clusters = [cluster_tl, cluster_tr, cluster_br, cluster_bl]

    return clusters

def IoU(mask: np.ndarray, corners: np.ndarray):
    """
    Computes the Intersection over Union (IoU) between a binary mask and a polygon defined by its corner points.

    Args:
        - mask (np.ndarray): segmentation mask of the receipt.
        - corners (np.ndarray): array containing the coordinates of the four corners of a polygon candidate.

    Returns:
        - iou (float): The IoU score, computed as the ratio of the intersection area to the union area.
    """
    try:
        corners = corners.reshape(4,2)
    except ValueError as e:
        raise ValueError(f"Cannot reshape array of size {corners.size} into shape (4,2).") from e

    corners = corners[:, np.newaxis, :]  # Add new axis because cv2.fillPoly wants inputs with shape (m, 1, n)
    polygon = cv2.fillPoly(np.zeros(mask.shape, dtype= np.uint8), pts= np.int32([corners]), color = 255)
    mask_bool = mask==255
    polygon_bool = polygon==255
    
    # Compute the area of intersection (only where both arrays have values of 255
    interArea = np.sum(np.logical_and(mask_bool, polygon_bool))

    # Compute the area of Union
    mask_area = np.sum(mask_bool)
    polygon_area = np.sum(polygon_bool)
    unionArea = mask_area + polygon_area - interArea

    # Compute iou
    iou = interArea / unionArea

    return iou

def resize_rectangle(rect, original_shape, new_shape):
    """
    Rescales a rectangle's coordinates to match a new image resolution.
    
    Args:
    - rect (np.ndarray): np.array of shape (4,2), the original rectangle's coordinates (x, y).
    - original_shape (tuple): (height, width) of the original image.
    - new_shape (tuple): (height, width) of the new image.

    Returns:
    - Rescaled rectangle coordinates as a np.array of shape (4,2).
    """
    orig_h, orig_w = original_shape
    new_h, new_w = new_shape

    # Compute scaling factors for x and y dimensions
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    # Apply scaling to each point in the rectangle
    scaled_rect = rect * np.array([scale_x, scale_y])

    return scaled_rect

def contour_to_rect(contour):
    pts = contour.reshape(-1, 2)
    rect = np.zeros_like(pts, dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def load_yolo_model(weights_path):
    """
    Loads a YOLO model based on the available hardware and format of the weights.
    
    - If the weights are PyTorch-based and a GPU is available, it loads a PyTorch model.
    - Otherwise, it converts the model to ONNX and loads it using ONNX Runtime for CPU acceleration.

    Args:
        - weights_path (str): Path to the YOLO weights file (.pt for PyTorch, .onnx for ONNX)
        
    Returns:
        - Loaded model (PyTorch or ONNX session)
    """
    
    def is_pytorch_model(path):
        """Check if the file is a PyTorch model"""
        return weights_path.endswith(".pt")

    def convert_to_onnx(pt_path, onnx_path="yolo_receipt_seg.onnx"):
        """Converts a PyTorch YOLO model to ONNX format."""
        model = YOLO(pt_path)
        model.export(format="onnx")  # Convert to ONNX
        del model
        return onnx_path

    if is_pytorch_model(weights_path):
        if torch.cuda.is_available():
            print("[INFO] Loading PyTorch YOLO model on GPU...")
            return YOLO(weights_path).to("cuda"), "pt"   # Load YOLO model on GPU
        else:
            print("[INFO] No GPU available, converting to ONNX for CPU acceleration...")
            weight_path = Path(weight_path)
            onnx_path = convert_to_onnx(weights_path, onnx_path=weights_path.with_suffix("onnx"))
            weights_path = onnx_path  # Update path to ONNX model

    if weights_path.endswith(".onnx"):
        print("[INFO] Loading ONNX YOLO model inference...")
        return YOLO(weights_path), "onnx"

    raise ValueError("Invalid model format: Only .pt (PyTorch) and .onnx (ONNX) are supported.")


