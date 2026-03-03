# PixelPure — Image Denoising Comparison

A complete image denoising pipeline that compares **traditional computer-vision scripting** (contrast stretch → median filter → morphological operations → adaptive threshold) against a **SwinUNet deep learning model** (Swin Transformer + U-Net architecture).

Includes **PixelPure**, a browser-based comparison website that runs both methods side-by-side using ONNX Runtime Web — no backend required.

---

## Repository Structure

```
Stage 1 - Image Denoising/
│
├── Python Scripting/
│   ├── Scripts/
│   │   ├── swinunet_model.py            # SwinUNet architecture (Swin Transformer + U-Net)
│   │   ├── swinunet_train.py            # Training script (AMP, gradient checkpointing)
│   │   ├── swinunet_test.py             # Tiled inference on test images
│   │   ├── swinunet_data.py             # Dataset & DataLoader with in-memory caching
│   │   ├── denoise_for_ocr.py           # Traditional denoising pipeline (OpenCV)
│   │   ├── synthetic_noise_add.py       # Generate synthetic noisy images from clean ones
│   │   ├── export_onnx.py               # Export trained model to ONNX for browser inference
│   │   ├── swinunet_explanation.txt      # Model architecture explanation
│   │   ├── swinunet_detailed_explanation.txt  # In-depth code walkthrough
│   │   ├── swinunet_steps.txt           # Step-by-step training/testing guide
│   │   └── traditional_vs_swinunet.txt  # Comparison of both approaches
│   │
│   └── Data/                            # (not included — see Data Setup below)
│       ├── Clean/                       # Clean ground-truth images
│       ├── Noisy/                       # Noisy input images (paired with Clean)
│       ├── Test/                        # Test images for inference
│       └── Denoised/                    # Output folder for denoised results
│
├── Comparison Website/
│   └── index.html                       # Browser-based comparison UI (Traditional vs SwinUNet)
│
├── .gitignore
└── README.md
```

> **Note:** Training/test images, model weights (`.pth`, `.onnx`), and virtual environments are excluded from this repository via `.gitignore`.

---

## Requirements

- **Python 3.11+**
- **PyTorch 2.x** with CUDA support (for GPU training)
- **ONNX & ONNXRuntime** (for export/verification)
- **OpenCV** (`opencv-python`)

### Install Dependencies

```bash
python -m venv .venv311
.venv311\Scripts\Activate.ps1          # Windows PowerShell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy Pillow onnx onnxruntime
```

---

## Usage

### 1. Prepare Data

Place your images in the `Python Scripting/Data/` folders:

| Folder   | Contents                                       |
|----------|-------------------------------------------------|
| `Clean/` | Clean ground-truth images                       |
| `Noisy/` | Corresponding noisy versions (same filenames)   |
| `Test/`  | Noisy images you want to denoise                |

You can generate synthetic noisy images using:
```bash
python "Python Scripting/Scripts/synthetic_noise_add.py"
```

### 2. Train the SwinUNet Model

```bash
python "Python Scripting/Scripts/swinunet_train.py"
```

- Uses paired `Noisy/` ↔ `Clean/` images
- Saves `best_swinunet.pth` (best validation PSNR) and `latest_swinunet.pth`
- GPU-accelerated with mixed precision (AMP)

### 3. Test / Denoise Images

```bash
python "Python Scripting/Scripts/swinunet_test.py"
```

- Loads `best_swinunet.pth` and denoises all images in `Data/Test/`
- Saves results to `Data/Denoised/`

### 4. Traditional Denoising (No Model Needed)

```bash
python "Python Scripting/Scripts/denoise_for_ocr.py"
```

Applies: contrast stretching → median filter → morphological open/close → adaptive thresholding.

### 5. Export to ONNX (For Browser Use)

```bash
python "Python Scripting/Scripts/export_onnx.py"
```

Exports the trained model to `Comparison Website/model.onnx` for browser inference.

### 6. Run the Comparison Website

```bash
cd "Comparison Website"
python -m http.server 8080
```

Then open **http://localhost:8080** in Chrome. The website:
- Auto-loads `model.onnx` from the same folder
- Lets you upload noisy images
- Runs **both** traditional and SwinUNet denoising in-browser
- Shows results side-by-side with download options

---

## Model Details

| Property               | Value                                      |
|------------------------|--------------------------------------------|
| Architecture           | SwinUNet (Swin Transformer + U-Net)        |
| Transformer Blocks     | 4 RSTB × 6 Swin Transformer Layers        |
| Embedding Dimension    | 96                                         |
| Parameters             | ~2.24M                                     |
| Input                  | Grayscale 256×256 (tiled for larger images)|
| Training Loss          | L1 + Edge Loss (text preservation)         |
| Best Validation PSNR   | 22.32 dB                                   |
| Best Validation SSIM   | 0.9434                                     |

---

## Documentation

Detailed explanations are included in the repository:

- [swinunet_steps.txt](Python%20Scripting/Scripts/swinunet_steps.txt) — Step-by-step guide to train and test
- [swinunet_explanation.txt](Python%20Scripting/Scripts/swinunet_explanation.txt) — Model architecture overview
- [swinunet_detailed_explanation.txt](Python%20Scripting/Scripts/swinunet_detailed_explanation.txt) — Full code walkthrough
- [traditional_vs_swinunet.txt](Python%20Scripting/Scripts/traditional_vs_swinunet.txt) — Comparison of both approaches
