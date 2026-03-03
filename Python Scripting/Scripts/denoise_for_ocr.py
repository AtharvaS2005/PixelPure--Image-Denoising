import os
import cv2
import numpy as np
from glob import glob

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
TEST_DIR = os.path.join(ROOT_DIR, 'Data', 'Test')
DENOISED_DIR = os.path.join(ROOT_DIR, 'Data', 'Denoised')
os.makedirs(DENOISED_DIR, exist_ok=True)

# Denoising pipeline

def enhance_contrast(img):
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def remove_small_noise(img):
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def fill_text_gaps(img):
    # Morphological closing to fill gaps in text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def adaptive_binarize(img):
    # Adaptive thresholding for robust binarization
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 25, 15)

def median_filter(img):
    # Median filter for salt-and-pepper noise
    return cv2.medianBlur(img, 3)

def inpaint_lines(img):
    # Inpaint horizontal/vertical white lines (cuts/erasures)
    mask = (img == 255).astype(np.uint8) * 255
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

# Process all images
image_paths = glob(os.path.join(TEST_DIR, '*'))
for img_path in image_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    # 1. Enhance contrast
    img = enhance_contrast(img)
    # 2. Median filter
    img = median_filter(img)
    # 3. Remove small noise
    img = remove_small_noise(img)
    # 4. Fill text gaps
    img = fill_text_gaps(img)
    # 5. Adaptive binarization
    img = adaptive_binarize(img)
    # 6. Inpaint lines/cuts (optional, can comment out if not needed)
    # img = inpaint_lines(img)
    # Save result
    base = os.path.basename(img_path)
    out_path = os.path.join(DENOISED_DIR, base)
    cv2.imwrite(out_path, img)

print(f"Processed {len(image_paths)} images. Denoised images saved to {DENOISED_DIR}.")
