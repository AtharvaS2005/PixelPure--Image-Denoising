"""
Export SwinUNet to ONNX for browser-based inference.
=====================================================
Usage:  python export_onnx.py
Output: ../../Comparison Website/model.onnx

The exported model accepts fixed-size 256x256 grayscale tiles
(float32, [0,1] range). The JavaScript frontend handles tiling.
"""

import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)

from swinunet_model import SwinUnet

MODEL_PTH = os.path.join(WORKSPACE, 'best_swinunet.pth')
OUT_DIR   = os.path.join(WORKSPACE, 'Comparison Website')
OUT_ONNX  = os.path.join(OUT_DIR, 'model.onnx')

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ──────────────────────────────────────────────────────
print("Loading SwinUNet weights ...", flush=True)
model = SwinUnet(
    img_size=128, in_chans=1, embed_dim=96,
    depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
    window_size=8, mlp_ratio=2.0,
)
state = torch.load(MODEL_PTH, map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()

# ── Export ──────────────────────────────────────────────────────────
print("Exporting to ONNX (opset 17, fixed 256x256) ...", flush=True)
dummy = torch.randn(1, 1, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        OUT_ONNX,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True,
    )

size_mb = os.path.getsize(OUT_ONNX) / (1024 ** 2)
print(f"Saved: {OUT_ONNX}  ({size_mb:.1f} MB)", flush=True)

# ── Optional verification ──────────────────────────────────────────
try:
    import onnx
    m = onnx.load(OUT_ONNX)
    onnx.checker.check_model(m)
    print("ONNX verification: PASSED", flush=True)
except ImportError:
    print("(onnx package not installed — skipping verification)", flush=True)
except Exception as e:
    print(f"ONNX verification warning: {e}", flush=True)

print("\nDone! Deploy index.html + model.onnx together.", flush=True)
