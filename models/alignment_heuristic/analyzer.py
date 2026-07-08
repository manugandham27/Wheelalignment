import cv2
import numpy as np
from utils.helper import load_config

def sort_corners(pts):
    """
    Sort 4 coordinates in order: top-left, top-right, bottom-right, bottom-left.
    """
    # Sum of coordinates: minimum is top-left, maximum is bottom-right
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def unwarp_tread_region(img, contour):
    """
    Find rotated bounding box around the tread contour,
    and apply perspective transform (homographic unwarping) to flatten it.
    """
    # Get rotated rectangle of the tread contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Sort the corners
    src_pts = sort_corners(box.astype(np.float32))
    
    # Target size for the unwarped rectangle
    width = 200
    height = 200
    
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Calculate projection matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # Apply homography
    unwarped = cv2.warpPerspective(img, matrix, (width, height))
    return unwarped, src_pts

def analyze_wear_asymmetry(image_path, config=None):
    """
    Classical CV module augmented with Otsu Segmentation and Homographic Unwarping.
    Projects curved tire tread perspective into a normalized flat rectangle
    prior to calculating left-edge vs center vs right-edge wear asymmetry.
    """
    if config is None:
        config = load_config()

    thresh_low = config["models"]["alignment_heuristic"].get("edge_low_threshold", 50)
    thresh_high = config["models"]["alignment_heuristic"].get("edge_high_threshold", 150)
    blur_k = config["models"]["alignment_heuristic"].get("gaussian_blur_ksize", 5)
    asymmetry_threshold = config["models"]["alignment_heuristic"].get("asymmetry_threshold", 0.15)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    h, w, _ = img.shape
    
    # 1. Image Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # 2. Otsu thresholding to segment the primary tire body
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    unwarped = None
    if contours:
        # Largest contour represents the tire face
        largest_contour = max(contours, key=cv2.contourArea)
        
        # If the contour area is reasonably large, apply unwarping
        if cv2.contourArea(largest_contour) > (h * w * 0.1):
            try:
                unwarped, src_pts = unwarp_tread_region(img, largest_contour)
            except Exception as e:
                print(f"Homographic unwarping failed, using original crop: {str(e)}")
                
    if unwarped is None:
        # Fallback to standard crop if segmentation or warping failed
        unwarped = img
        
    # Process the unwarped/flattened tread image
    gray_unwarped = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
    blurred_unwarped = cv2.GaussianBlur(gray_unwarped, (blur_k, blur_k), 0)
    edges = cv2.Canny(blurred_unwarped, thresh_low, thresh_high)
    
    uw_h, uw_w = edges.shape
    margin = int(uw_w * 0.05)
    usable_w = uw_w - 2 * margin
    
    left_bound = margin + int(usable_w * 0.33)
    right_bound = margin + int(usable_w * 0.66)
    
    left_zone = edges[:, margin:left_bound]
    center_zone = edges[:, left_bound:right_bound]
    right_zone = edges[:, right_bound:(uw_w - margin)]

    density_left = float(np.mean(left_zone) / 255.0)
    density_center = float(np.mean(center_zone) / 255.0)
    density_right = float(np.mean(right_zone) / 255.0)

    avg_density = (density_left + density_right) / 2.0
    if avg_density > 0:
        asymmetry_score = abs(density_left - density_right) / (avg_density + 1e-6)
    else:
        asymmetry_score = 0.0

    misalignment_flag = asymmetry_score > asymmetry_threshold
    explanations = []
    confidence = min(1.0, asymmetry_score / (asymmetry_threshold * 2.0))
    
    if misalignment_flag:
        if density_left < density_right:
            explanations.append("Excessive wear detected on the LEFT edge. Possible toe-out or negative camber misalignment.")
        else:
            explanations.append("Excessive wear detected on the RIGHT edge. Possible toe-in or positive camber misalignment.")
    else:
        shoulder_avg = (density_left + density_right) / 2.0
        if density_center < (shoulder_avg * 0.75):
            explanations.append("Center tread is more worn than shoulders, suggesting chronic over-inflation.")
        elif shoulder_avg < (density_center * 0.75):
            explanations.append("Shoulder treads are more worn than the center, suggesting chronic under-inflation.")
        else:
            explanations.append("Tread wear appears symmetric and normal. Alignment is likely within limits.")

    return {
        "asymmetry_score": asymmetry_score,
        "density_left": density_left,
        "density_center": density_center,
        "density_right": density_right,
        "alignment_flag": bool(misalignment_flag),
        "alignment_confidence": float(confidence),
        "diagnosis": " ".join(explanations)
    }
