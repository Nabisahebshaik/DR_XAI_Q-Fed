"""
Q-FedSecure DR-XAI: Image Preprocessing & Quality Assessment (IQA) Module
Designed for rural healthcare field triage (ASHA/ANM workers and PHC centers).
"""

import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Union


def assess_image_quality(
    image_input: Union[str, np.ndarray],
    blur_threshold: float = 100.0,
    low_light_threshold: float = 45.0,
    over_exposure_threshold: float = 215.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Performs real-time Image Quality Assessment (IQA) on fundus photographs
    to filter out motion blur, out-of-focus artifacts, and poor illumination
    common in low-cost portable ophthalmoscopes.

    Parameters:
        image_input: Filepath string or uint8 BGR/RGB numpy array.
        blur_threshold: Minimum variance of Laplacian required for sharpness.
        low_light_threshold: Minimum mean luminance required.
        over_exposure_threshold: Maximum mean luminance before flagging glare.

    Returns:
        is_gradeable (bool): True if image passes quality checks for AI grading.
        metrics (dict): Detailed diagnostic measurements and human-readable feedback.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return False, {"error": f"File not found: {image_input}", "status": "File Error"}
        img = cv2.imread(image_input)
        if img is None:
            return False, {"error": "Invalid image format", "status": "Decode Error"}
    else:
        img = image_input.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale & isolate green channel (highest retinal contrast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    green_channel = img[:, :, 1] if len(img.shape) == 3 else gray

    # 1. Laplacian Variance for Sharpness / Blur Detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    # 2. Illumination & Exposure Analysis
    mean_intensity = float(np.mean(green_channel))
    std_intensity = float(np.std(green_channel))

    # 3. Retinal Field of View (FOV) Mask Detection
    # Filter background border to measure actual retinal disk coverage
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    fov_ratio = float(np.count_nonzero(mask) / (gray.shape[0] * gray.shape[1]))

    # Quality Criteria Evaluation
    reasons = []
    is_gradeable = True

    if laplacian_var < blur_threshold:
        is_gradeable = False
        reasons.append(f"Excessive Blur/Defocus (Score: {laplacian_var:.1f} < {blur_threshold})")

    if mean_intensity < low_light_threshold:
        is_gradeable = False
        reasons.append(f"Severe Under-Exposure / Low Light (Mean: {mean_intensity:.1f} < {low_light_threshold})")
    elif mean_intensity > over_exposure_threshold:
        is_gradeable = False
        reasons.append(f"Over-Exposure / Glare (Mean: {mean_intensity:.1f} > {over_exposure_threshold})")

    if fov_ratio < 0.25:
        is_gradeable = False
        reasons.append(f"Insufficient Retinal Coverage (FOV Area: {fov_ratio*100:.1f}%)")

    status_message = "Gradeable (High Quality)" if is_gradeable else "Rejected: " + "; ".join(reasons)

    metrics = {
        "is_gradeable": is_gradeable,
        "laplacian_variance": round(laplacian_var, 2),
        "mean_intensity": round(mean_intensity, 2),
        "std_intensity": round(std_intensity, 2),
        "fov_coverage_ratio": round(fov_ratio, 3),
        "status": status_message,
        "reasons": reasons
    }

    return is_gradeable, metrics


def crop_retina_circle(image: np.ndarray, tolerance: int = 15) -> np.ndarray:
    """
    Crops the fundus image to the circular retinal region, removing excess
    black borders and centering the optic disk / macula region.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mask = gray > tolerance
    if not np.any(mask):
        return image

    # Find bounding box of valid retina
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Ensure square crop
    h = rmax - rmin
    w = cmax - cmin
    max_dim = max(h, w)
    cy, cx = (rmin + rmax) // 2, (cmin + cmax) // 2

    rmin_new = max(0, cy - max_dim // 2)
    rmax_new = min(image.shape[0], cy + max_dim // 2)
    cmin_new = max(0, cx - max_dim // 2)
    cmax_new = min(image.shape[1], cx + max_dim // 2)

    return image[rmin_new:rmax_new, cmin_new:cmax_new]


def apply_clahe_enhancement(
    image_input: Union[str, np.ndarray],
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
    target_size: Tuple[int, int] = (224, 224)
) -> Union[np.ndarray, None]:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel
    in LAB color space to enhance micro-aneurysms, hemorrhages, and hard exudates
    under inconsistent rural field lighting.

    Parameters:
        image_input: Filepath or numpy image (BGR or RGB).
        clip_limit: Threshold for contrast limiting in CLAHE.
        tile_grid_size: Grid size for local histogram equalization.
        target_size: Output resolution for DenseNet201 (default: 224x224).

    Returns:
        Preprocessed and normalized float32 image array in range [0.0, 1.0].
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return None
        img = cv2.imread(image_input)
        if img is None:
            return None
        # Convert BGR from OpenCV to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_input.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 1. Circular Crop to remove peripheral camera borders
    img_cropped = crop_retina_circle(img)

    # 2. Resize to canonical model resolution
    img_resized = cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_AREA)

    # 3. Convert to LAB Color Space for Luminance Equalization
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 4. Apply CLAHE on L (Luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # 5. Merge channels back and convert to RGB
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # 6. Apply mild Gaussian smoothing to suppress sensor noise in low-cost lenses
    rgb_smoothed = cv2.GaussianBlur(rgb_enhanced, (3, 3), 0.5)

    # 7. Normalize pixel values to [0, 1] for neural network input
    normalized = rgb_smoothed.astype(np.float32) / 255.0

    return normalized


def preprocess_for_inference(image_input: Union[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Combined production pipeline: validates quality, applies CLAHE, and formats
    tensor for QCNET model inference.
    """
    is_valid, metrics = assess_image_quality(image_input)
    processed = apply_clahe_enhancement(image_input)
    return processed, metrics
