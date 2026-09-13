"""
Q-FedSecure DR-XAI: Advanced Retinal Vessel Segmentation & Micro-Lesion Masking
Uses Morphological Top-Hat Transform, Adaptive Thresholding, and Contour
Geometry Filtering to cleanly separate continuous tubular blood vessels from
isolated circular dot hemorrhages and microaneurysms.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Union
from src.preprocessing import crop_retina_circle


def segment_retinal_vessels(
    image_input: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (512, 512),
    min_vessel_area: int = 15,
    min_aspect_ratio: float = 2.0,
    max_circularity_for_dots: float = 0.70
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Performs morphology-based vascular tree segmentation:
    1. Extracts Green channel (optimal contrast) and applies CLAHE normalization.
    2. Executes Multi-Scale Morphological Top-Hat Transforms (White Top-Hat on inverted channel /
       Black Top-Hat on green channel) to isolate tubular ridge structures.
    3. Employs adaptive thresholding to detect candidate vascular pixels.
    4. Executes Contour Geometric Filtering:
       - Retains elongated, continuous tubular branches (aspect ratio >= 2.0).
       - Discards compact, circular objects (microaneurysms, dot hemorrhages).
    5. Computes accurate vascular density and produces a fluorescent cyan overlay.

    Parameters:
        image_input: File path or RGB/BGR NumPy array.
        target_size: Canvas resolution (default: 512x512).
        min_vessel_area: Minimum pixel area to consider.
        min_aspect_ratio: Minimum major/minor axis ratio for tubular structures.
        max_circularity_for_dots: Threshold to reject round dot lesions.

    Returns:
        vessel_mask (np.ndarray): Clean binary vessel mask (0 or 255).
        vessel_overlay (np.ndarray): RGB visualization with highlighted cyan vessels.
        vessel_density (float): Retinal vessel density percentage within FOV.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_input.copy()
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

    # 1. Circular Retinal Crop & Green Channel Isolation
    img_cropped = crop_retina_circle(img)
    img_resized = cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_AREA)
    green = img_resized[:, :, 1]

    # 2. Retinal Field of View (FOV) Mask
    _, fov_mask = cv2.threshold(green, 15, 255, cv2.THRESH_BINARY)
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fov_mask = cv2.erode(fov_mask, kernel_circle, iterations=1)

    # 3. CLAHE Normalization on Green Channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green)

    # 4. Multi-Scale Morphological Top-Hat Transform
    # Black Top-Hat on green channel extracts dark tubular vessels against brighter retina
    kernel_fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_coarse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    bth_fine = cv2.morphologyEx(green_clahe, cv2.MORPH_BLACKHAT, kernel_fine)
    bth_coarse = cv2.morphologyEx(green_clahe, cv2.MORPH_BLACKHAT, kernel_coarse)
    vessels_enhanced = cv2.add(bth_fine, bth_coarse)

    # 5. Background Subtraction & Adaptive Thresholding
    vessels_enhanced = cv2.bitwise_and(vessels_enhanced, vessels_enhanced, mask=fov_mask)
    
    # Adaptive threshold combined with lower-bound intensity threshold
    thresh_adaptive = cv2.adaptiveThreshold(
        vessels_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, -3
    )
    _, thresh_fixed = cv2.threshold(vessels_enhanced, 14, 255, cv2.THRESH_BINARY)
    raw_binary = cv2.bitwise_and(thresh_adaptive, thresh_fixed)
    raw_binary = cv2.bitwise_and(raw_binary, raw_binary, mask=fov_mask)

    # 6. Morphological Contour Geometry Filtering (Reject Dot Hemorrhages & Microaneurysms)
    clean_vessel_mask = np.zeros_like(raw_binary)
    contours, _ = cv2.findContours(raw_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_vessel_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Circularity: 4 * pi * Area / Perimeter^2 (Dot lesions have high circularity ~0.7-1.0)
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        # Minimum bounding box for aspect ratio / elongation
        rect = cv2.minAreaRect(cnt)
        (_, (width, height), _) = rect
        major_axis = max(width, height)
        minor_axis = max(1.0, min(width, height))
        aspect_ratio = major_axis / minor_axis

        # Filter criteria: Keep large contiguous vascular trees OR elongated branches
        is_large_vessel = area >= 50
        is_tubular = (aspect_ratio >= min_aspect_ratio) and (circularity < max_circularity_for_dots)

        if is_large_vessel or is_tubular:
            cv2.drawContours(clean_vessel_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Morphological closing to bridge fine capillary gaps
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_vessel_mask = cv2.morphologyEx(clean_vessel_mask, cv2.MORPH_CLOSE, kernel_bridge)
    clean_vessel_mask = cv2.bitwise_and(clean_vessel_mask, clean_vessel_mask, mask=fov_mask)

    # 7. Vessel Density Calculation within Retinal FOV
    fov_pixel_count = np.count_nonzero(fov_mask)
    vessel_pixel_count = np.count_nonzero(clean_vessel_mask)
    vessel_density = float((vessel_pixel_count / max(1, fov_pixel_count)) * 100.0)

    # 8. Create Fluorescent Diagnostic Overlay (Bright Cyan vessels on fundus backdrop)
    vessel_overlay = img_resized.copy()
    vessel_overlay[clean_vessel_mask == 255] = [0, 255, 220]

    return clean_vessel_mask, vessel_overlay, round(vessel_density, 2)


def detect_lesion_candidates(
    image_input: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (512, 512)
) -> Dict[str, Any]:
    """
    Extracts candidate regions for Diabetic Retinopathy pathology:
    - Red Lesions (Microaneurysms, Dot/Blot Hemorrhages)
    - Bright Lesions (Hard Exudates, Cotton Wool Spots)
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_input.copy()
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

    img_cropped = crop_retina_circle(img)
    img_resized = cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_AREA)

    # Color space conversions
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    green = img_resized[:, :, 1]

    _, fov_mask = cv2.threshold(green, 15, 255, cv2.THRESH_BINARY)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    fov_mask = cv2.erode(fov_mask, kernel_erode)

    # 1. Hard Exudates (Bright yellow/white spots in Lab L & B channels)
    l_channel = lab[:, :, 0]
    b_channel = lab[:, :, 2]
    bright_mask = (l_channel > 165) & (b_channel > 135) & (fov_mask > 0)
    bright_lesions = (bright_mask * 255).astype(np.uint8)

    # 2. Red Lesions (Dark red / brown spots in Green channel below median)
    median_green = np.median(green[fov_mask > 0]) if np.any(fov_mask > 0) else 100
    red_mask = (green < (median_green * 0.55)) & (fov_mask > 0)
    red_lesions = (red_mask * 255).astype(np.uint8)

    # 3. Create Diagnostic Overlay
    lesion_overlay = img_resized.copy()
    lesion_overlay[bright_lesions == 255] = [255, 235, 59]  # Yellow for exudates
    lesion_overlay[red_lesions == 255] = [255, 30, 30]      # Crimson red for hemorrhages

    return {
        "bright_lesions_count": int(np.count_nonzero(bright_lesions)),
        "red_lesions_count": int(np.count_nonzero(red_lesions)),
        "lesion_overlay": lesion_overlay,
        "exudates_mask": bright_lesions,
        "hemorrhages_mask": red_lesions
    }
