"""
Data Loading and Preprocessing for SwinUNet Denoising
=====================================================
- Matches clean/noisy image pairs by filename
- Extracts random patches for training (with augmentation)
- Center-crop patches for validation (deterministic)
- Normalizes to [0, 1] (no mean/std normalization — needed for residual learning)
- Supports index-based subset selection for train/val split
- Configurable repeat factor for more patches per epoch
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class DenoisingDataset(Dataset):
    """Paired clean/noisy image dataset with random patch extraction.

    Args:
        clean_dir:  Path to clean images.
        noisy_dir:  Path to noisy images.
        patch_size: Square patch size to extract (default 128).
        augment:    Enable random flips & 90-deg rotations.
        indices:    Optional list of file indices to use (for train/val split).
        repeat:     Virtual dataset multiplier — each image appears `repeat`
                    times per epoch with different random crops.
        cache:      Cache all images in RAM on first access (huge speedup).
    """

    def __init__(self, clean_dir, noisy_dir, patch_size=128,
                 augment=True, indices=None, repeat=1, cache=True):
        super().__init__()
        self.patch_size = patch_size
        self.augment = augment
        self.repeat = max(1, repeat)
        self.use_cache = cache

        # Match clean <-> noisy by filename
        clean_files = set(os.listdir(clean_dir))
        noisy_files = set(os.listdir(noisy_dir))
        common = sorted(clean_files & noisy_files)

        if len(common) == 0:
            raise ValueError(
                f"No matching filenames between\n  {clean_dir}\n  {noisy_dir}")

        if indices is not None:
            common = [common[i] for i in indices]

        self.clean_paths = [os.path.join(clean_dir, f) for f in common]
        self.noisy_paths = [os.path.join(noisy_dir, f) for f in common]

        # In-memory cache: read all images once at init
        self._clean_cache = [None] * len(self.clean_paths)
        self._noisy_cache = [None] * len(self.noisy_paths)

        if self.use_cache:
            print(f"  Caching {len(common)} image pairs into RAM ...", end=' ', flush=True)
            for i in range(len(self.clean_paths)):
                c = cv2.imread(self.clean_paths[i], cv2.IMREAD_GRAYSCALE)
                n = cv2.imread(self.noisy_paths[i], cv2.IMREAD_GRAYSCALE)
                if c is None or n is None:
                    raise IOError(f"Cannot load image pair: {self.clean_paths[i]}")
                # Store as uint8 to save RAM (~8x vs float32)
                self._clean_cache[i] = c
                self._noisy_cache[i] = n
            print("done.")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.clean_paths) * self.repeat

    # ------------------------------------------------------------------
    def _load(self, idx):
        """Return (clean, noisy) as float32 [0,1] arrays, from cache or disk."""
        if self.use_cache and self._clean_cache[idx] is not None:
            return (self._clean_cache[idx].astype(np.float32) / 255.0,
                    self._noisy_cache[idx].astype(np.float32) / 255.0)

        clean = cv2.imread(self.clean_paths[idx], cv2.IMREAD_GRAYSCALE)
        noisy = cv2.imread(self.noisy_paths[idx], cv2.IMREAD_GRAYSCALE)
        if clean is None or noisy is None:
            raise IOError(f"Cannot load image pair at index {idx}: "
                          f"{self.clean_paths[idx]}")
        clean = clean.astype(np.float32) / 255.0
        noisy = noisy.astype(np.float32) / 255.0

        if self.use_cache:
            self._clean_cache[idx] = clean
            self._noisy_cache[idx] = noisy
        return clean, noisy

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        real_idx = idx % len(self.clean_paths)

        clean, noisy = self._load(real_idx)

        H, W = clean.shape
        ps = self.patch_size

        # --- Crop ---
        if self.augment:
            top  = np.random.randint(0, max(1, H - ps + 1))
            left = np.random.randint(0, max(1, W - ps + 1))
        else:
            top  = (H - ps) // 2
            left = (W - ps) // 2

        clean_p = clean[top:top + ps, left:left + ps]
        noisy_p = noisy[top:top + ps, left:left + ps]

        # --- Augmentation ---
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                clean_p = np.fliplr(clean_p).copy()
                noisy_p = np.fliplr(noisy_p).copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                clean_p = np.flipud(clean_p).copy()
                noisy_p = np.flipud(noisy_p).copy()
            # Random 90-degree rotation (0/1/2/3 times)
            k = np.random.randint(0, 4)
            if k > 0:
                clean_p = np.rot90(clean_p, k).copy()
                noisy_p = np.rot90(noisy_p, k).copy()

        # (1, H, W) tensors
        clean_t = torch.from_numpy(clean_p).unsqueeze(0)
        noisy_t = torch.from_numpy(noisy_p).unsqueeze(0)

        return noisy_t, clean_t


# =====================================================================
#  Default Paths (relative to this script)
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CLEAN_DIR  = os.path.join(ROOT_DIR, 'Data', 'Clean')
NOISY_DIR  = os.path.join(ROOT_DIR, 'Data', 'Noisy')
