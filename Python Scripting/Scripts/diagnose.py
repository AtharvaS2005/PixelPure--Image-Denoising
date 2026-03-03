"""Quick diagnostic to verify the AMP fix works."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swinunet_model import SwinUnet
from swinunet_data import DenoisingDataset, CLEAN_DIR, NOISY_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# --- Load one batch ---
ds = DenoisingDataset(CLEAN_DIR, NOISY_DIR, patch_size=128,
                      augment=False, indices=[0,1,2,3], repeat=1)
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=4)
noisy, clean = next(iter(dl))
noisy, clean = noisy.to(device), clean.to(device)

print(f"Noisy: mean={noisy.mean():.4f}  Clean: mean={clean.mean():.4f}")

# --- Import the FIXED CombinedLoss ---
from swinunet_train import CombinedLoss, calc_ssim

model = SwinUnet(img_size=128, in_chans=1, embed_dim=96,
                 depths=(6,6,6,6), num_heads=(6,6,6,6),
                 window_size=8, mlp_ratio=2.0).to(device)
for rstb in model.rstb_layers:
    rstb.use_checkpoint = True

criterion = CombinedLoss(ssim_weight=0.15, edge_weight=0.05).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scaler = torch.amp.GradScaler('cuda')

print("\\n--- 5 training steps with AMP (model forward in autocast, loss in fp32) ---")
for step in range(5):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # Forward in AMP
    with torch.amp.autocast('cuda'):
        out = model(noisy)
    
    # Loss in float32 (OUTSIDE autocast)
    loss = criterion(out, clean)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    has_nan = any(p.grad.isnan().any() for p in model.parameters() if p.grad is not None)
    scaler.step(optimizer)
    scaler.update()
    
    with torch.no_grad():
        mse = torch.mean((out.float() - clean) ** 2)
        psnr = 10.0 * torch.log10(1.0 / mse)
    
    print(f"  Step {step+1}: loss={loss.item():.5f}  psnr={psnr.item():.2f}dB  "
          f"grad_norm={grad_norm:.3f}  NaN={has_nan}  scale={scaler.get_scale():.0f}")

print("\\nIf NaN=False and PSNR is improving, the fix works!")
