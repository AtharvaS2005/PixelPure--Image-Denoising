"""
SwinUNet Training Script for Image Denoising
=============================================
Features:
  - Charbonnier (smooth-L1) + SSIM combined loss
  - Cosine-annealing LR schedule
  - Automatic Mixed Precision (AMP) on CUDA
  - Gradient clipping for stability
  - Saves BOTH best_swinunet.pth AND latest_swinunet.pth every epoch
  - Train / validation split (by image, no data leakage)
  - Comprehensive per-epoch metrics: Loss, PSNR, SSIM
  - Early stopping with configurable patience
"""

import os
import sys
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from swinunet_model import SwinUnet
from swinunet_data import DenoisingDataset, CLEAN_DIR, NOISY_DIR

# Suppress harmless LR scheduler warning
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*')


# =====================================================================
#  Paths
# =====================================================================
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
BEST_MODEL     = os.path.join(WORKSPACE_ROOT, 'best_swinunet.pth')
LATEST_MODEL   = os.path.join(WORKSPACE_ROOT, 'latest_swinunet.pth')


# =====================================================================
#  Metrics
# =====================================================================

def calc_psnr(pred, target, data_range=1.0):
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:
        return torch.tensor(100.0)
    return 20.0 * torch.log10(torch.tensor(data_range, device=mse.device)) \
           - 10.0 * torch.log10(mse)


def _gauss_window(window_size, sigma, device):
    coords = torch.arange(window_size, dtype=torch.float32,
                           device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)


def calc_ssim(pred, target, window_size=11, data_range=1.0):
    """Structural Similarity Index (single-channel)."""
    ch = pred.size(1)
    win = _gauss_window(window_size, 1.5, pred.device).expand(ch, -1, -1, -1)
    pad = window_size // 2

    mu1     = F.conv2d(pred,   win, padding=pad, groups=ch)
    mu2     = F.conv2d(target, win, padding=pad, groups=ch)
    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu12    = mu1 * mu2
    s1_sq   = F.conv2d(pred * pred,     win, padding=pad, groups=ch) - mu1_sq
    s2_sq   = F.conv2d(target * target,  win, padding=pad, groups=ch) - mu2_sq
    s12     = F.conv2d(pred * target,    win, padding=pad, groups=ch) - mu12

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
    return ssim_map.mean()


# =====================================================================
#  Loss Functions
# =====================================================================

class CharbonnierLoss(nn.Module):
    """Charbonnier loss — smooth approximation of L1."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class EdgeLoss(nn.Module):
    """Sobel-based edge loss — penalises erasing text/edges.

    Computes horizontal and vertical Sobel gradients of both predicted
    and target images, then applies Charbonnier loss between them.
    This forces the model to preserve sharp edges (= text strokes).
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps2 = eps ** 2
        # Sobel kernels  (1, 1, 3, 3)  — not trainable
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)
        self.register_buffer('kx', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('ky', sobel_y.view(1, 1, 3, 3))

    def _gradient(self, img):
        gx = F.conv2d(img, self.kx, padding=1)
        gy = F.conv2d(img, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, pred, target):
        diff = self._gradient(pred) - self._gradient(target)
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class CombinedLoss(nn.Module):
    """Charbonnier + SSIM + Edge loss for text-preserving denoising.

    All computations forced to float32 to avoid AMP float16 overflow.
    """

    def __init__(self, ssim_weight=0.15, edge_weight=0.05, window_size=11):
        super().__init__()
        self.char = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.ssim_w = ssim_weight
        self.edge_w = edge_weight
        self.win_size = window_size

    def forward(self, pred, target):
        # Force float32 to prevent NaN from AMP overflow
        pred32 = pred.float()
        tgt32  = target.float()

        l_char = self.char(pred32, tgt32)
        l_ssim = self.ssim_w * (1.0 - calc_ssim(pred32, tgt32, self.win_size))
        l_edge = self.edge_w * self.edge(pred32, tgt32)
        return l_char + l_ssim + l_edge


# =====================================================================
#  Main
# =====================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"VRAM   : {mem_gb:.1f} GB")

    # ---- hyperparameters ------------------------------------------------
    PATCH_SIZE = 128
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    LR         = 2e-4
    WEIGHT_DEC = 1e-4
    GRAD_CLIP  = 1.0
    VAL_RATIO  = 0.1
    PATIENCE   = 20
    REPEAT     = 8      # patches per image per epoch

    # ---- model ----------------------------------------------------------
    model = SwinUnet(
        img_size=PATCH_SIZE, in_chans=1, embed_dim=96,
        depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
        window_size=8, mlp_ratio=2.0,
    ).to(device)

    # Enable gradient checkpointing to save VRAM (mandatory for 8GB card)
    for rstb in model.rstb_layers:
        rstb.use_checkpoint = True

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_params:,} total  |  {n_train:,} trainable")

    # ---- dataset split (by image index, zero data-leakage) ---------------
    all_files = sorted(
        set(os.listdir(CLEAN_DIR)) & set(os.listdir(NOISY_DIR)))
    total = len(all_files)
    val_cnt   = max(1, int(total * VAL_RATIO))
    train_cnt = total - val_cnt

    g    = torch.Generator().manual_seed(42)
    perm = torch.randperm(total, generator=g).tolist()
    train_idx = perm[:train_cnt]
    val_idx   = perm[train_cnt:]

    train_ds = DenoisingDataset(
        CLEAN_DIR, NOISY_DIR, patch_size=PATCH_SIZE,
        augment=True, indices=train_idx, repeat=REPEAT)
    val_ds = DenoisingDataset(
        CLEAN_DIR, NOISY_DIR, patch_size=PATCH_SIZE,
        augment=False, indices=val_idx, repeat=1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0,
                              pin_memory=True, drop_last=True,
                              persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0,
                              pin_memory=True,
                              persistent_workers=False)

    print(f"Train  : {len(train_ds)} patches  ({train_cnt} images x{REPEAT})")
    print(f"Val    : {len(val_ds)} patches  ({val_cnt} images)")
    print(f"Patch  : {PATCH_SIZE}x{PATCH_SIZE}   Batch: {BATCH_SIZE}")

    # ---- resume from checkpoint if available ----------------------------
    if os.path.exists(BEST_MODEL):
        print(f"Resuming from checkpoint: {BEST_MODEL}")
        model.load_state_dict(torch.load(BEST_MODEL, map_location=device,
                                         weights_only=True))
    else:
        print("Starting from scratch (no checkpoint found).")

    # ---- optimizer & scheduler ------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DEC, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ---- loss -----------------------------------------------------------
    # ssim_weight=0.15 : structure/text preservation
    # edge_weight=0.05 : Sobel gradient loss to prevent text erasure
    # All loss computations forced to float32 to avoid AMP overflow
    criterion = CombinedLoss(ssim_weight=0.15, edge_weight=0.05).to(device)

    # ---- AMP scaler -----------------------------------------------------
    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- training loop --------------------------------------------------
    best_psnr = 0.0
    wait = 0

    hdr = (f"{'Ep':>4} | {'TrLoss':>9} | {'TrPSNR':>9} | "
           f"{'VaLoss':>9} | {'VaPSNR':>9} | {'VaSSIM':>8} | "
           f"{'LR':>10} | {'Time':>6}")
    sep = '=' * len(hdr)

    print(f"\n{sep}\n{hdr}\n{sep}", flush=True)

    total_batches = len(train_loader)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ---- train ------------------------------------------------------
        model.train()
        t_loss, t_psnr, t_n = 0.0, 0.0, 0

        for batch_i, (noisy, clean) in enumerate(train_loader, 1):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(noisy)

            # Loss computed in float32 (outside autocast) to avoid NaN
            loss = criterion(out, clean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            bs = noisy.size(0)
            with torch.no_grad():
                t_loss += loss.item() * bs
                t_psnr += calc_psnr(out.clamp(0, 1), clean).item() * bs
                t_n    += bs

            # progress bar every 10 batches
            if batch_i % 10 == 0 or batch_i == total_batches:
                elapsed = time.time() - t0
                pct = 100.0 * batch_i / total_batches
                print(f"\r  Ep {epoch:>3}/{NUM_EPOCHS}  "
                      f"[{batch_i:>4}/{total_batches}] "
                      f"{pct:5.1f}%  "
                      f"loss={t_loss/t_n:.5f}  "
                      f"psnr={t_psnr/t_n:.2f}dB  "
                      f"{elapsed:.0f}s", end='', flush=True)

        print()   # newline after progress bar
        t_loss /= t_n
        t_psnr /= t_n

        # ---- validate ---------------------------------------------------
        model.eval()
        v_loss, v_psnr, v_ssim, v_n = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    out  = model(noisy)

                # Loss in float32 (outside autocast)
                loss = criterion(out, clean)

                oc = out.clamp(0, 1)
                bs = noisy.size(0)
                v_loss += loss.item() * bs
                v_psnr += calc_psnr(oc, clean).item() * bs
                v_ssim += calc_ssim(oc, clean).item() * bs
                v_n    += bs

        v_loss /= v_n
        v_psnr /= v_n
        v_ssim /= v_n

        scheduler.step()

        # ---- save models ------------------------------------------------
        torch.save(model.state_dict(), LATEST_MODEL)   # always

        tag = ''
        if v_psnr > best_psnr:
            best_psnr = v_psnr
            wait = 0
            torch.save(model.state_dict(), BEST_MODEL)
            tag = ' *BEST'
        else:
            wait += 1

        lr_now = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        print(f"{epoch:>4} | {t_loss:>9.6f} | {t_psnr:>7.2f}dB | "
              f"{v_loss:>9.6f} | {v_psnr:>7.2f}dB | {v_ssim:>7.5f} | "
              f"{lr_now:>10.2e} | {dt:>5.1f}s{tag}", flush=True)

        if wait >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).", flush=True)
            break

    # ---- summary --------------------------------------------------------
    print(f"\n{sep}", flush=True)
    print(f"Training complete!  Best Val PSNR: {best_psnr:.2f} dB", flush=True)
    print(f"  Best model   -> {BEST_MODEL}", flush=True)
    print(f"  Latest model -> {LATEST_MODEL}", flush=True)


if __name__ == '__main__':
    main()
