# Satellite Image Super-Resolution Pipeline 🛰️

> Bridging the Resolution Gap: Sentinel-2 (10m/pixel) → 8x Super-Resolution mimicking WorldView/SPOT commercial quality

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/GPU-Kaggle%20T4%2FP100-teal)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Problem Statement

Publicly available satellite imagery from the European Space Agency's **Sentinel-2**
constellation is free and globally available, but limited to **10 meters/pixel** — a resolution
at which cars, small buildings, and road markings are invisible. Commercial alternatives
(Maxar WorldView, Airbus SPOT 6/7) offer sub-1.5m clarity but are **financially prohibitive**
for researchers and humanitarian organizations.

This project uses Deep Learning to **synthetically bridge this resolution gap** — generating
commercial-grade 8x super-resolved imagery from free Sentinel-2 inputs, without
hallucinating features that do not exist in the original scene.

---

## Architecture: 3-Tier Hybrid Ensemble

```
LR Input (Sentinel-2, 10m/px)
         │
         ├──► [Path A] Swin2SR (Vision Transformer) ──► Fidelity Output
         │         (Geometric accuracy, no hallucinations)
         │
         ├──► [Path B] Real-ESRGAN (Generative Model) ──► Perceptual Output
         │         (Texture sharpness, urban clarity)
         │
         └──► [Adaptive Fusion Engine]
                   alpha * Fidelity + (1-alpha) * Perceptual
                   (alpha = high for vegetation/water, low for urban grids)
                         │
                         ▼
              SR Output (8x, ~1.25m/px equivalent)
```

### Hallucination Guardrail
The model is constrained by two physics-aware loss functions:
- **Cycle Consistency Loss**: SR output, when downsampled back to LR, must reconstruct the original input.
- **Gradient Profile Loss**: Spatial edge derivatives of SR must match HR ground truth.

---

## Project Status

| Stage | Description | Status |
|---|---|---|
| Pre-Stage 0 | Scaffolding, GEE auth, Kaggle setup | 🔄 In Progress |
| Stage 1 | Swin2SR baseline inference (8x) | ✅ Complete |
| Stage 2 | GEE data pipeline, normalization, metrics | ✅ Complete |
| Stage 3 | Fine-tuning on WorldStrat | ✅ Complete |
| Stage 4 | Hallucination guardrail loss functions | ⬜ Not Started |
| Stage 5 | 3-Tier ensemble + Streamlit UI | ⬜ Not Started |

---

## Tech Stack

| Component | Technology |
|---|---|
| Core DL | PyTorch 2.1+ |
| SR Model (Fidelity) | Swin2SR (Vision Transformer via `timm`) |
| SR Model (Perception) | Real-ESRGAN |
| Data Streaming | Google Earth Engine Python API |
| Geospatial Processing | Rasterio, NumPy stride tricks |
| Metrics | scikit-image (PSNR, SSIM) + custom Hallucination Score |
| GPU Compute | Kaggle (NVIDIA T4 / P100, free tier) |
| Visualization | Streamlit + streamlit-image-comparison |

---

## Dataset

**WorldStrat** — ~10,000 km² of globally stratified, paired LR/HR imagery:
- **LR**: Sentinel-2 (10m/pixel, 16-bit, multispectral)
- **HR**: Airbus SPOT 6/7 (1.5m/pixel)
- Diverse land-use: urban, agricultural, forests, humanitarian sites

**Normalization**: Raw 16-bit values clipped at `3000.0` and divided by `3000.0` → `float32`.
This preserves high-frequency urban gradients that are crushed by naive `/65535` normalization.

---

## Repository Structure

```
satellite-super-resolution/
├── src/
│   ├── model_swin2sr.py       # Swin2SR model wrapper (8x upscale)
│   ├── model_ensemble.py      # 3-Tier hybrid ensemble + adaptive fusion
│   ├── data_loader.py         # GEE streaming + PyTorch Dataset
│   ├── loss_functions.py      # Cycle consistency + Gradient profile losses
│   ├── training_loop.py       # Fine-tuning loop with checkpointing
│   ├── metrics.py             # PSNR, SSIM, Hallucination Score
│   └── utils_memory.py        # Rasterio tiling + Gaussian blending
├── notebooks/
│   ├── stage1_inference.ipynb # Baseline SR inference demo
│   └── main_training.ipynb    # Full training entry point (Kaggle)
├── configs/
│   ├── hyperparams.yaml       # Global config
│   └── stage3_hyperparams.yaml
├── weights/                   # (git-ignored) .pth checkpoints
├── data/                      # (git-ignored) local test patches
├── outputs/                   # (git-ignored) SR output images
├── requirements.txt
└── PROJECT_JOURNEY.md         # Decision log + interview notes
```

---

## Quick Start (Inference)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate GEE
earthengine authenticate

# 3. Run Stage 1 baseline inference
jupyter notebook notebooks/stage1_inference.ipynb
```

---

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| PSNR | Peak Signal-to-Noise Ratio vs. HR reference | > 30 dB |
| SSIM | Structural Similarity Index | > 0.85 |
| Hallucination Score | `1 - SSIM(Downsample(SR), LR)` | < 0.05 |

---

*Built with [Antigravity IDE](https://antigravity.dev) · Mentor: Antigravity AI*
