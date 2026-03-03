"""
SwinUNet Testing / Inference Script
====================================
Features:
  - Tiled inference with overlap blending for full-resolution images
  - Computes PSNR & SSIM when ground truth (Clean) is available
  - Supports both 'best' and 'latest' models via command-line arg
  - Outputs denoised images to Data/Denoised/

Usage:
    python swinunet_test.py            # uses best model (default)
    python swinunet_test.py best       # explicit best
    python swinunet_test.py latest     # use latest checkpoint
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from swinunet_model import SwinUnet


# =====================================================================
#  Paths
# =====================================================================
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR       = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TEST_DIR       = os.path.join(ROOT_DIR, 'Data', 'Test')
CLEAN_DIR      = os.path.join(ROOT_DIR, 'Data', 'Clean')
OUT_DIR        = os.path.join(ROOT_DIR, 'Data', 'Denoised')
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
#  Numpy Metrics
# =====================================================================

def psnr_np(img1, img2, data_range=255.0):
    """PSNR between two uint8 numpy arrays."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def ssim_np(img1, img2, data_range=255.0):
    """SSIM between two grayscale uint8 numpy arrays."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    mu1   = cv2.filter2D(i1, -1, window)[5:-5, 5:-5]
    mu2   = cv2.filter2D(i2, -1, window)[5:-5, 5:-5]
    mu1sq = mu1 ** 2
    mu2sq = mu2 ** 2
    mu12  = mu1 * mu2

    s1sq = cv2.filter2D(i1 ** 2,  -1, window)[5:-5, 5:-5] - mu1sq
    s2sq = cv2.filter2D(i2 ** 2,  -1, window)[5:-5, 5:-5] - mu2sq
    s12  = cv2.filter2D(i1 * i2,  -1, window)[5:-5, 5:-5] - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1sq + mu2sq + C1) * (s1sq + s2sq + C2))
    return float(ssim_map.mean())


# =====================================================================
#  Tiled Inference
# =====================================================================

def denoise_full(model, img_gray, device, tile_size=256, overlap=32):
    """Denoise a full-resolution grayscale image via tiled inference.

    Args:
        model:     Trained SwinUnet (eval mode).
        img_gray:  (H, W) uint8 numpy array.
        device:    torch device.
        tile_size: Processing tile edge length.
        overlap:   Pixels of overlap between adjacent tiles.

    Returns:
        (H, W) uint8 denoised numpy array.
    """
    H, W = img_gray.shape

    # Small image → process in one shot
    if H <= tile_size and W <= tile_size:
        t = torch.from_numpy(img_gray.astype(np.float32) / 255.0)
        t = t.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
        return np.clip(out.squeeze().cpu().numpy() * 255.0,
                       0, 255).astype(np.uint8)

    # Tiled inference with uniform overlap averaging
    stride = tile_size - overlap
    output = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_st  = max(0, y_end - tile_size)
            x_st  = max(0, x_end - tile_size)

            tile = img_gray[y_st:y_end, x_st:x_end]
            th, tw = tile.shape

            # Pad if smaller than tile_size
            if th < tile_size or tw < tile_size:
                padded = np.full((tile_size, tile_size),
                                 fill_value=128, dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            t = torch.from_numpy(tile.astype(np.float32) / 255.0)
            t = t.unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                out_t = model(t)

            out_np = out_t.squeeze().cpu().numpy()[:th, :tw]

            output[y_st:y_end, x_st:x_end] += out_np
            weight[y_st:y_end, x_st:x_end] += 1.0

    output /= np.maximum(weight, 1e-8)
    return np.clip(output * 255.0, 0, 255).astype(np.uint8)


# =====================================================================
#  Main
# =====================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- model choice ---------------------------------------------------
    choice = 'best'
    if len(sys.argv) > 1 and sys.argv[1] in ('best', 'latest'):
        choice = sys.argv[1]

    model_path = os.path.join(WORKSPACE_ROOT, f'{choice}_swinunet.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Train first with:  python swinunet_train.py")
        return

    # ---- load model -----------------------------------------------------
    model = SwinUnet(
        img_size=128, in_chans=1, embed_dim=96,
        depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
        window_size=8, mlp_ratio=2.0,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Loaded '{choice}' model from {model_path}")

    # ---- gather test images ---------------------------------------------
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    test_files = sorted(
        f for f in os.listdir(TEST_DIR)
        if os.path.splitext(f)[1].lower() in exts)

    if not test_files:
        print("No test images found.")
        return

    psnrs, ssims = [], []

    print(f"\nProcessing {len(test_files)} images ...")
    print(f"{'File':<30} {'Size':>12}   {'PSNR':>10}   {'SSIM':>10}")
    print('-' * 70)

    for fname in test_files:
        img = cv2.imread(os.path.join(TEST_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  SKIP  {fname}  (cannot load)")
            continue

        H, W = img.shape
        t0 = time.time()

        denoised = denoise_full(model, img, device,
                                tile_size=256, overlap=32)

        dt = time.time() - t0
        cv2.imwrite(os.path.join(OUT_DIR, fname), denoised)

        # Metrics (if ground truth exists)
        gt_path = os.path.join(CLEAN_DIR, fname)
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None and gt.shape == denoised.shape:
                p = psnr_np(denoised, gt)
                s = ssim_np(denoised, gt)
                psnrs.append(p)
                ssims.append(s)
                print(f"  {fname:<28} {H}x{W:>5}   "
                      f"{p:>8.2f} dB   {s:>9.5f}   ({dt:.1f}s)")
                continue

        print(f"  {fname:<28} {H}x{W:>5}       N/A           N/A   ({dt:.1f}s)")

    # ---- summary --------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Denoised images saved to: {OUT_DIR}")
    if psnrs:
        print(f"Avg PSNR : {np.mean(psnrs):.2f} dB  "
              f"({len(psnrs)} images w/ ground truth)")
        print(f"Avg SSIM : {np.mean(ssims):.5f}")
    print(f"Total    : {len(test_files)} images processed")


if __name__ == '__main__':
    main()
