# Import packages
import os
import numpy as np
import ultralytics
from roboflow import Roboflow
from ultralytics import YOLO
from abc import ABC, abstractmethod
from scipy.fftpack import dct, idct
from sklearn import cluster
from skimage import morphology, filters, transform
from skimage.segmentation import clear_border
import utils
import itertools
import cv2



class BaseReceiptScanner(ABC):
    @abstractmethod
    def process_receipt(self, img_path):
        """
        Full processing pipeline for receipt scanning.
        
        Args:
            - img_path (str): Path to input image
            
        Returns:
            - dict: Processing results containing original image, scanned receipt, and enhanced version
        """
        pass


# TODO: Upload custom yolo model to hf to automatically load it from there if no path is specified
class CustomReceiptScanner(BaseReceiptScanner):
    def __init__(self, config_path=None, model_path=None):
        """
        Initialise custom receipt scanner with YOLO model trained on instance segmentation.
        
        Args:
            - config_path (str): Path to the config file containing important variables for the methods, e.g. thresholds
            - model_path (str): Path to the trained YOLO weights
        """
        # Load onnx model if only cpu is available to accelerated inference on CPUs
        self.model, self.model_type = utils.load_yolo_model(model_path)           

        # Load config file
        if config_path:
            config = utils.load_yaml_config(config_path)
        else:
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Construct the path to the YAML file
            config_path = os.path.join(current_dir, "config.yaml")
            config = utils.load_yaml_config(config_path)
            print(f"No path for custom config file was given. Loading standard config file from {config_path}.")

        # Load variables from config file
        self.sigmaX, self.sigmaY = config.get("sigmaX"), config.get("sigmaY")  # for cv2.GaussianBlur()
        self.threshold1, self.threshold2 = config.get("threshold1"), config.get("threshold2")  # for cv2.Canny
        self.quantile = config.get("quantile")  # for cluster.estimate_bandwidth
        self.window_size, self.k = config.get("window_size"), config.get("k")  # for Sauvola threshold

    def segment_receipt(self, img_path):
        """
        Segment receipt from image using YOLO model
        
        Args:
            image_path (str): Path to input image
            confidence (float): Confidence threshold
            
        Returns:
            tuple (np.ndarray, np.ndarray): segmentation mask of receipt in original size and on extended background
        """
        if self.model_type=="pt":
            result = self.model.predict(source=img_path, imgsz=640)
        else:
            result = self.model(source=img_path, imgsz=640)  # different call argument for onnx model
        mask = result[0].masks.data[0].cpu().numpy().astype(np.uint8)*255
        mask_extended = utils.extend_segmentation_mask(mask)

        return mask, mask_extended

    def _find_contours(self, mask_extended):
        """
        Get contours of (extended) segmentation mask of receipt by applying Gaussian blurring to remove noise, finding
        the edges using Canny edge detection and at last dilating the edges. Additionally, we get the convex hull of
        the found edge smooth out the possibly crumpled contour (this will make fitting lines to the edges way easier later)

        Args:
            - mask_extended (np.ndarray): The extended segmentation mask of the receipt from YOLO

        Returns:
            - convex_hull (np.ndarray): An array containing the points of the convex Hull of the largest found contour
        """
        
        blurred = cv2.GaussianBlur(mask_extended, (self.sigmaX, self.sigmaY), 0)
        edge = cv2.Canny(blurred, self.threshold1, self.threshold2)
        dilated = cv2.dilate(edge, kernel=morphology.disk(1))

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest contour (assume that this will be the receipt)
        largest_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]

        # Get convex Hull
        convex_hull = cv2.convexHull(largest_contour)

        return convex_hull

    def _get_corner_combinations(self, convex_hull, mask_extended):
        """
        Find the corners of the receipt to use for perspective warp in the next preprocessing step by fitting lines to the convex
        hull of the found receipt and then determining the corner candidates by choosing the combination of four corner candidates whose
        polygon has the highes iou with the segmentation mask of the receipt.

        Args:
            - convex_hull (np.ndarray): An array containing the points of the convex Hull of the largest found contour
            - mask_extended_shape (tuple): Height and width of the extended segmentation mask

        Returns:
            - 
        """

        # Fit lines to the edges of the convex hull
        height, width = mask_extended.shape
        convex_hull_filled = cv2.fillPoly(np.zeros((height, width), dtype= np.uint8), pts= [convex_hull], color = 255)
        blurred = cv2.GaussianBlur(convex_hull_filled, (self.sigmaX, self.sigmaY), 0)
        edge = cv2.Canny(blurred, self.threshold1, self.threshold2)
        lines = np.array(transform.probabilistic_hough_line(edge))  # Use probabilistic Hough Lines
        angles = np.array([np.abs(np.atan2(a[1]-b[1], a[0]-b[0]) - np.pi/2) for a,b in lines])  # Determine angles of lines

        # Categorise lines into vertical and horizontal lines by their angles
        verticalLines = lines[angles < np.pi/4] 
        horizontalLines = lines[angles >= np.pi/4]

        # Get all intersections of fitted hough lines and cluster them to reduce the amount of points
        intersections = np.array([utils.intersection(utils.line(vl[0],vl[1]), utils.line(hl[0], hl[1])) for vl in verticalLines for hl in horizontalLines])
        
        bw = cluster.estimate_bandwidth(intersections, quantile=self.quantile)
        corners = cluster.MeanShift(bandwidth=bw).fit(intersections).cluster_centers_
        rect = utils.contour_to_rect(corners)[:4]

        # Cluster all points into tl, tr, bl, br clusters to reduce amount of combinations as we will only pick one tl/tr/bl/br candidate
        # to build a polygon (this reduces the amount of combinations from binom_coeff(n_corners 4) to n_tl * n_tr * n_bl * n_br
        clusters = utils.get_corners_cluster(rect=rect, corners=corners)

        # Get all possible valid combinations of corners
        valid_corner_combinations = list(itertools.product(*clusters))

        return valid_corner_combinations

    def _get_corners(self, corner_combinations, mask, mask_extended, img):
        """
        Loop over all corner combinations and calculate IoU with the segmentation mask 
        to get the corner candidates for which the IoU is maximal.
        
        Parameters:
         - corner_combinations (list[tuple]): List of tuples containing corner candidates
         - mask (np.ndarray): The 2D array of the segmentation mask of the receipt

         Returns:
             - corners (np.ndarray): Array containing the corners of the receipt in the original image
        """
        iou_tmp = 0
        best_candidate = None  
        
        for comb in corner_combinations:
            iou_comb = utils.IoU(mask, np.array(comb))
            
            if iou_comb > iou_tmp:
                best_candidate = comb
                iou_tmp = iou_comb  # Update best IoU value

        # Compute mask dimensions which keep the aspect ratio of the image, e.g. the onnx model always returns masks of shape (640,640) which does not keep
        # the aspect ratio of the original image. Therefore We need to account for this now 
        img_height, img_width, img_channels = img.shape
        mask_height, mask_width = mask.shape
        mask_width_corrected = int((img_width / img_height) * mask_height)

        crns = np.array(best_candidate)
        crns = utils.transform_corners(crns, mask_extended.shape, (mask_height, mask_width_corrected))  # Resize corners to fit original masks dimensions
        # Resize corners to match original dimensions of the image (also use the corrected mask width here!)
        corners = utils.resize_rectangle(crns, (mask_height, mask_width_corrected), (img_height, img_width))

        return corners

    def _warp_perspective(self, img, rect):
        # Unpack the rect points (in the order: tl, tr, br, bl)
        rect = utils.contour_to_rect(rect)
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        
        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
        # Take the maximum of the width and height values to get the final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        
        # Define the destination points (ordered as: tl, tr, br, bl)
        dst = np.array([
            [0, 0],  # top-left
            [maxWidth - 1, 0],  # top-right
            [maxWidth - 1, maxHeight - 1],  # bottom-right
            [0, maxHeight - 1]  # bottom-left
        ], dtype="float32")
        
        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Warp the perspective to grab the screen
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    def _enhance_receipt(self, receipt):
        """
        Enhance receipt image using sharpening and contrast adjustment
        Args:
            receipt (np.ndarray): Input receipt image
        Returns:
            np.array: Enhanced receipt
        """
        
        # Apply adaptive thresholding (Sauvola)
        receipt_mask = receipt < filters.threshold_sauvola(receipt, window_size=self.window_size, k=self.k)
        
        # Remove artifacts at the border
        receipt_mask = clear_border(receipt_mask)
        
        # Apply mask to suppress background noise
        receipt = 1 - (1 - receipt) * receipt_mask
        
        # Apply contrast enhancement
        receipt = receipt**3
        
        # Convert to uint8 for visualization
        receipt = (receipt*255).astype(np.uint8)
        
        # Apply threshold
        receipt[receipt<245] = 0   

        return receipt

        
    def _normalize_global_illumination(self, gray):
        """
        Normalize global illumination by removing slow illumination gradients using a high-pass filter.
        As slow gradients correspond to low frequency modulation of the image intensity, filtering the image
        with a high-pass filter should even out global illumination

        Args:
            - gray (np.ndarray): The gray image of the receipt
            
        """
        frequencies = dct(dct(gray, axis=0), axis=1)
 
        frequencies[:2,:2] = 0
         
        gray = idct(idct(frequencies, axis=1), axis=0)
         
        gray = (gray - gray.min()) / (gray.max() - gray.min()) # renormalize to range [0:1]

        return gray

    def process_receipt(self, img_path):
        """
        Full processing pipeline for receipt scanning
        Args:
            img_path (str): Path to input image
        Returns:
            dict: Processing results containing original image, scanned receipt, and enhanced version
        """
        # Read image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Segment receipt
        mask, mask_extended = self.segment_receipt(img_path)
        
        # Get corners
        convex_hull = self._find_contours(mask_extended)
        corner_combinations = self._get_corner_combinations(convex_hull, mask_extended)
        receipt_corners = self._get_corners(corner_combinations, mask, mask_extended, img)

        
        
        if receipt_corners is None:
            raise ValueError("No receipt detected in image")
            
        # Transform perspective
        warped = self._warp_perspective(gray, receipt_corners)
        
        # Enhance image
        enhanced = self._enhance_receipt(warped)
        
        return {
            'original': img,
            'mask': mask,
            'mask_extended': mask_extended,
            'warped': warped,
            'enhanced': enhanced,
            'corners': receipt_corners
        }
        
        

        
        
        

    