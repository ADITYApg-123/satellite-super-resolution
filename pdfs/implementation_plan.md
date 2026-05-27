# Satellite Super-Resolution: Master Implementation Roadmap

A 5-Stage, step-by-step build plan for the **3-Tier Hybrid Ensemble SR Pipeline**
(Sentinel-2 LR → 4x/8x HR output mimicking WorldView/SPOT quality).

> **Workflow Legend**
> - 📝 = Update `PROJECT_JOURNEY.md`
> - 📖 = Update `README.md`
> - 💾 = Git commit checkpoint
> - 🎓 = Mentor explanation / tutorial step (IDE-external action required)
> - ✅ = Aditya confirms completion before moving forward

---

## Pre-Stage 0: Project Scaffolding & Environment Setup

> **Goal**: Establish the full directory skeleton, install all dependencies, and connect external platforms (GEE, Kaggle) **before writing a single line of model code**. A broken environment is the #1 cause of wasted hours.

### Step 0.1 — Create the Full Directory Structure

Create all folders and placeholder files so the IDE can index them:

```
satellite-super-resolution/
├── src/
│   ├── model_swin2sr.py          # Stage 1 — Swin2SR model wrapper
│   ├── model_ensemble.py         # Stage 5 — 3-Tier hybrid ensemble
│   ├── data_loader.py            # Stage 2 — GEE streaming + PyTorch Dataset
│   ├── loss_functions.py         # Stage 4 — Cycle consistency + Gradient profile
│   ├── training_loop.py          # Stage 3 — Fine-tuning loop
│   ├── metrics.py                # Stage 2 — PSNR, SSIM, Hallucination Score
│   └── utils_memory.py           # Stage 2/5 — Rasterio tiling + Gaussian blending
├── notebooks/
│   ├── stage1_inference.ipynb    # Stage 1 — Run baseline Swin2SR inference
│   └── main_training.ipynb       # Stage 3+ — Full training entry point
├── configs/
│   ├── hyperparams.yaml          # Global config (LR, batch size, patch size, etc.)
│   └── stage3_hyperparams.yaml   # Stage 3 specific overrides
├── weights/                      # Downloaded .pth model weights (git-ignored)
├── data/                         # Local sample test patches (git-ignored)
├── outputs/                      # Saved SR output images (git-ignored)
├── kernel-metadata.json          # Kaggle API deployment config
├── requirements.txt              # All pip dependencies
├── PROJECT_JOURNEY.md            # Interview log (our story)
└── README.md                     # Project documentation (always current)
```

- **Action**: Run a single `mkdir` / `New-Item` command to create all folders at once.
- **Why**: Monolithic scripts in a flat directory break when pushed to Kaggle cloud. Modularity from day one prevents path chaos later.

---

### Step 0.2 — Create `requirements.txt`

Pin all dependencies upfront so local and Kaggle environments are **identical**:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Core DL framework |
| `timm` | Provides Swin Transformer building blocks |
| `earthengine-api` | GEE Python API for data streaming |
| `rasterio` | Geospatial raster windowing (out-of-core memory) |
| `numpy` | Array math, stride tricks |
| `opencv-python` | Image pre/post-processing |
| `scikit-image` | PSNR and SSIM metrics |
| `pyyaml` | Load `hyperparams.yaml` configs |
| `tqdm` | Training progress bars |
| `streamlit` | Evaluation web UI |
| `streamlit-image-comparison` | Interactive before/after slider |
| `matplotlib` | Metric plots |
| `kaggle` | Kaggle CLI for cloud deployment |

- **Why `timm`?**: Swin2SR's building blocks (`WindowAttention`, `PatchEmbed`) are included in `timm`. This prevents us from having to manually clone the Swin2SR repo and manage its internal imports.

---

### Step 0.3 — 🎓 Google Earth Engine (GEE) Authentication Tutorial

> **This cannot be done inside the IDE. Aditya must do this manually.**

**What is GEE?** Google Earth Engine is a cloud-based planetary-scale geospatial analysis platform. We stream real Sentinel-2 satellite tiles directly from it instead of downloading terabytes locally.

**Steps**:
1. Go to [https://earthengine.google.com](https://earthengine.google.com) → Sign up with your Google account (free for non-commercial research).
2. Wait for approval (usually instant for student accounts).
3. Once approved, open a terminal in the project root and run:
   ```
   earthengine authenticate
   ```
4. A browser window will open. Log in with the same Google account. Copy the authorization code and paste it back into the terminal.
5. A credential file is saved to `~/.config/earthengine/credentials`.
6. Verify it works: `python -c "import ee; ee.Initialize(); print('GEE connected!')"`.

**Why not just download the WorldStrat dataset?** The raw WorldStrat dataset is ~250 GB. GEE's `ee.data.computePixels` streams exactly the patch we need, at training time, directly into PyTorch memory — no storage required.

---

### Step 0.4 — 🎓 Kaggle API Setup Tutorial

> **This cannot be done inside the IDE. Aditya must do this manually.**

**What is Kaggle here?** We will use Kaggle's free NVIDIA T4/P100 GPU (30 hrs/week) for all heavy model training. We push code from this IDE → Kaggle, train, pull checkpoints back.

**Steps**:
1. Go to [https://www.kaggle.com](https://www.kaggle.com) → Account → Settings → API → "Create New Token".
2. A `kaggle.json` file is downloaded. Place it at: `C:\Users\adity\.kaggle\kaggle.json`.
3. In a terminal: `pip install kaggle` then `kaggle datasets list` to verify it works.
4. Run `kaggle kernels init -p .` in the project root to generate `kernel-metadata.json`.
5. Edit `kernel-metadata.json`:
   - `"enable_gpu": true`
   - `"enable_internet": true`
   - `"kernel_type": "notebook"`
   - `"code_file": "notebooks/main_training.ipynb"`

**Why Kaggle over Google Colab?** Kaggle gives 30 hrs/week of free GPU with **persistent storage** and a proper CLI push/pull workflow. Colab sessions disconnect randomly.

---

### Step 0.5 — Create Initial `PROJECT_JOURNEY.md` and `README.md`

- Create the first entry in `PROJECT_JOURNEY.md`: "Project initialized. Directory structure created. GEE and Kaggle configured."
- Create a starter `README.md` with project title, problem statement, and architecture overview.

### 💾 Commit Checkpoint 0: "chore: project scaffold, requirements, and platform setup"

---

## Stage 1: Minimal Working SR — Swin2SR Baseline Inference

> **Goal**: Get a real, working super-resolution output on a single test image using pre-trained weights. No training yet. This proves the pipeline is structurally sound end-to-end.

### Step 1.1 — Download Solafune Pre-Trained Weights

- **What**: Download the pre-trained Swin2SR `.pth` checkpoint from the Solafune competition repository (Hugging Face or GitHub release). Place it in `/weights/`.
- **Why Solafune?**: The Solafune Team N weights were trained specifically on satellite imagery — not natural photos. Using them gives us a geometrically stable baseline immediately, rather than starting from ImageNet weights.
- **Why not train from scratch?**: Training a Vision Transformer from scratch requires millions of image pairs and weeks of compute. Pre-trained weights = thousands of hours of GPU time donated to us for free.

### Step 1.2 — Build `/src/model_swin2sr.py`

A single clean Python module with two functions:
- `initialize_swin2sr_model(weight_path)`: Instantiates the `Swin2SR` architecture with the exact hyperparameters matching the Solafune checkpoint (`embed_dim=180`, `depths=[6,6,6,6,6,6]`, `num_heads=[6,6,6,6,6,6]`, `window_size=8`, `upscale=4`) and loads the state dict in evaluation mode.
- `run_inference(model, input_tensor)`: Runs a single forward pass inside `torch.no_grad()` for memory efficiency.

**Why `torch.no_grad()`?** During inference, PyTorch normally allocates memory to store gradients for potential backpropagation. Since we are not training here, `no_grad()` cuts VRAM usage roughly in half.

### Step 1.3 — Build `/notebooks/stage1_inference.ipynb`

A step-by-step Jupyter notebook that:
1. Loads a single sample Sentinel-2 test patch (a small `.tif` or `.png` file we create manually).
2. Calls `normalize_sentinel_radiometry` (clip at 3000, divide by 3000) to prepare the tensor.
3. Passes it through `initialize_swin2sr_model` → `run_inference`.
4. Saves and displays the LR input vs. SR output side-by-side.
5. Prints PSNR and SSIM values vs. the original (if a HR reference is available).

### Step 1.4 — Visual Sanity Check

- Do roads look sharper? Do building edges appear? Does the image look like a real place?
- If SR output looks blurry or corrupted → the weight mismatch is the most likely cause (wrong `upscale` factor or wrong `embed_dim`).

### 📝 Journey Log Entry: "Stage 1 Complete: Achieved first super-resolved output with Swin2SR + Solafune weights. Key learning: weight hyperparams must exactly match the checkpoint architecture."
### 📖 Update README: Add "Stage 1: Inference Baseline" section with sample output image.
### 💾 Commit Checkpoint 1: "feat: stage 1 — Swin2SR baseline inference script and notebook"

---

## Stage 2: The Data Pipeline — GEE Streaming, Normalization & Metrics

> **Goal**: Build the production-grade PyTorch DataLoader that streams real, paired LR/HR patches from Google Earth Engine on-demand. No more static test images.

### Step 2.1 — Build the GEE Streaming Logic (inside `data_loader.py`)

- **What**: A function that accepts a bounding box (lat/lon), calls `ee.data.computePixels`, and returns a raw NumPy array of Sentinel-2 bands.
- **Key Design**: Implement **retry logic with exponential backoff**. GEE throttles concurrent API calls. Without retry logic, your DataLoader will crash during long training runs.
- **Why `computePixels` and not `Export`?**: GEE's `Export` functions are asynchronous — they send data to Google Drive (minutes of latency). `computePixels` is synchronous and streams pixels directly into RAM (milliseconds).

### Step 2.2 — Build the Radiometric Normalization Function (inside `data_loader.py`)

- **What**: `normalize_sentinel_radiometry(raw_array)`:
  - Clip all pixel values at `3000.0` (the ceiling).
  - Divide by `3000.0`.
  - Cast to `float32`.
- **Why clip at 3000, not 65535?**: Sentinel-2 is 16-bit (max value = 65535). But physically meaningful land surface reflectance values are clustered between 0 and ~3000. Dividing by 65535 would compress all real variation into a tiny slice of the 0-1 range, destroying the high-frequency urban gradients the model needs to learn.

### Step 2.3 — Build the Stride-Trick Patch Extractor (inside `utils_memory.py`)

- **What**: `extract_strided_windows(master_array, tile_size, overlap)` using `numpy.lib.stride_tricks.as_strided`.
- **Why stride tricks?**: `as_strided` creates a *view* of the array (no memory copy). For a 10,000×10,000 px satellite image, copying overlapping patches would require ~gigabytes of RAM. Stride tricks do it in kilobytes.
- **Why overlapping patches?**: Without overlap, tile boundaries produce seam artifacts in the final stitched image. Overlap + Gaussian blending (Stage 5) eliminates those seams.

### Step 2.4 — Build the Cloud Mask Handler

- **What**: A utility that reads the Sentinel-2 `QA60` band (bitmask layer) to detect and skip cloud-contaminated patches.
- **Why**: Cloud-covered patches look like uniform white blobs. Training on them teaches the model to output white blobs for normal land — a catastrophic hallucination.

### Step 2.5 — Build `/src/metrics.py`

Three evaluation functions:
- `compute_psnr(sr_image, hr_image)`: Peak Signal-to-Noise Ratio. Higher is better (>30 dB is good).
- `compute_ssim(sr_image, hr_image)`: Structural Similarity Index. Range 0-1, >0.85 is good.
- `compute_hallucination_score(sr_image, lr_image, scale_factor)`: Our custom guardrail metric. Downsample the SR output back to LR resolution and compute `1.0 - SSIM` against the original LR. If this score is high, the model invented details that cannot be traced back to the input.

### Step 2.6 — Build the PyTorch `WorldStratDataset` class

- A `torch.utils.data.Dataset` subclass wrapping the GEE streamer.
- `__getitem__` returns `(lr_tensor, hr_tensor)` pairs, fully normalized.
- `__len__` returns the number of WorldStrat bounding boxes.

### 📝 Journey Log Entry: "Stage 2 Complete: GEE streaming pipeline working. Key learnings: 3000-clip normalization, stride tricks for memory efficiency, importance of cloud masks."
### 📖 Update README: Add "Stage 2: Data Pipeline" section with normalization rationale.
### 💾 Commit Checkpoint 2: "feat: stage 2 — GEE data loader, normalization, metrics, memory utilities"

---

## Stage 3: Fine-Tuning Swin2SR on WorldStrat

> **Goal**: Move from inference-only to a full training loop. Adapt the Swin2SR model to our specific Sentinel-2 → SPOT 6/7 domain with properly tuned hyperparameters.

### Step 3.1 — Create `/configs/stage3_hyperparams.yaml`

Define all training hyperparameters in a YAML config (never hardcode in scripts):

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | `2e-4` | Conservative start; Vision Transformers are sensitive to LR |
| `lr_scheduler` | `CosineAnnealingLR` | Smooth LR decay, avoids sharp drops that kill fine-tuning |
| `warmup_epochs` | `5` | Prevents large gradient updates from corrupting pre-trained weights in early epochs |
| `batch_size` | `4` | Small batch for T4 GPU VRAM budget |
| `patch_size` | `128` (LR) → `512` (HR) | At 4x scale factor |
| `loss` | `L1` | L1 preserves median sharpness; L2 averages → blurry output |
| `gradient_clip` | `1.0` | Prevents exploding gradients in attention layers |
| `epochs` | `100` | With checkpointing every 10 epochs |

### Step 3.2 — Build `/src/training_loop.py`

Functions:
- `execute_training_step(model, optimizer, lr_tensor, hr_tensor)`: Single forward pass, L1 loss, backward, optimizer step.
- `run_validation_epoch(model, val_loader)`: Compute PSNR/SSIM on held-out patches.
- `save_checkpoint(model, optimizer, epoch, path)`: Save `.pth` checkpoint.
- `load_checkpoint(model, optimizer, path)`: Resume from checkpoint (critical for cloud training interruptions).

**Why checkpoint every 10 epochs?** Kaggle sessions can time out. Without periodic checkpointing, a 12-hour training run is lost completely on disconnect.

### Step 3.3 — Build `/notebooks/main_training.ipynb`

The Kaggle-compatible entry point that:
1. Authenticates GEE.
2. Instantiates `WorldStratDataset` and `DataLoader`.
3. Loads model from Solafune weights.
4. Runs the training loop with live PSNR/SSIM logging.
5. Saves the final fine-tuned checkpoint.

### Step 3.4 — 🎓 Kaggle Push Tutorial

Once the notebook is ready locally:
```
kaggle kernels push -p .
kaggle kernels status <your-username>/satellite-super-resolution
kaggle kernels output <your-username>/satellite-super-resolution
```
Pull back the trained `.pth` file and place it in `/weights/swin2sr_finetuned.pth`.

### 📝 Journey Log Entry: "Stage 3 Complete: Fine-tuned Swin2SR on WorldStrat. Key learning: L1 > L2 for SR. Cosine LR schedule + warmup critical. Checkpoint every 10 epochs."
### 📖 Update README: Add training metrics (PSNR/SSIM before vs. after fine-tuning).
### 💾 Commit Checkpoint 3: "feat: stage 3 — training loop, hyperparams config, Kaggle notebook"

---

## Stage 4: Hallucination Guardrails — Physics-Aware Loss Functions

> **Goal**: Prevent the model from inventing geospatial features. This is the single most important differentiator for the 10-pt Hallucination Guardrail score.

### Step 4.1 — Build `/src/loss_functions.py`

**Function 1: `downsampling_consistency_loss(sr_tensor, lr_tensor, scale_factor)`**
- **Logic**: Take the SR output → bicubic downsample by `1/scale_factor` → compare with original LR using L1.
- **Why**: If the model truly recovered real structure (not hallucinated it), then degrading the SR image back to LR resolution must reproduce the original LR input. If it doesn't match, the model invented something.
- **Formula**: `L_cycle = ||Bicubic_Downsample(SR, 1/s) - LR||₁`

**Function 2: `gradient_profile_loss(sr_tensor, hr_tensor)`**
- **Logic**: Compute first-order spatial gradients (x and y) for both SR and HR. Take the L1 difference of those gradients.
- **Why**: Pixel-level L1 loss penalizes wrong intensity but not wrong sharpness. Gradient loss directly penalizes blurry edges — the most common visual artifact in SR models.
- **Formula**: `L_grad = ||∂SR/∂x - ∂HR/∂x||₁ + ||∂SR/∂y - ∂HR/∂y||₁`

**The Composite Loss Function**:
```
L_total = λ₁·L_pixel + λ₂·L_cycle + λ₃·L_gradient
```
Where `λ₁=1.0`, `λ₂=0.1`, `λ₃=0.05` (starting weights, to be tuned).

### Step 4.2 — Integrate Composite Loss into `training_loop.py`

Replace the simple `F.l1_loss` call with the composite loss. Monitor each term separately in logs so we can detect if one term is dominating.

### Step 4.3 — NaN/Exploding Gradient Monitor

Add assertions to the training loop:
- After every loss computation: `assert not torch.isnan(loss)`.
- Check gradient norms per layer. Flag if any norm exceeds `10.0`.

### 📝 Journey Log Entry: "Stage 4 Complete: Physics guardrails implemented. Key learning: Cycle consistency = self-supervised constraint requiring no extra labels. Gradient loss = sharpness enforcer."
### 📖 Update README: Add section on hallucination guardrail architecture.
### 💾 Commit Checkpoint 4: "feat: stage 4 — composite loss functions (cycle consistency + gradient profile)"

---

## Stage 5: Perceptual Enhancement — The 3-Tier Hybrid Ensemble

> **Goal**: Layer Real-ESRGAN on top of Swin2SR and build the adaptive fusion engine. This is what transforms a "good model" into a "stunning model" for the Eye Test (20 pts).

### Step 5.1 — Integrate Real-ESRGAN (Perception Path B)

- Load Real-ESRGAN with **Satlas pre-trained weights** (satellite-specific, avoids natural-image over-sharpening).
- Wrap it in an identical inference interface to Swin2SR for plug-and-play fusion.

### Step 5.2 — Build the Adaptive Fusion Engine (inside `model_ensemble.py`)

**`adaptive_ensemble_fusion(fidelity_tensor, perception_tensor, alpha_map)`**

The `alpha_map` is a per-pixel weight matrix (same spatial dimensions as the SR output):
- **High alpha** (→1.0, trusts Swin2SR): Dense vegetation, water bodies, forests — hallucination-sensitive regions.
- **Low alpha** (→0.0, trusts Real-ESRGAN): Urban grids, road networks, structured buildings — sharpness-critical regions.

**How is `alpha_map` generated?** From a lightweight semantic segmentation pass on the LR input (e.g., a pre-trained lightweight SegFormer or even a simple NDVI index to separate vegetation vs. built-up land).

### Step 5.3 — Build the Gaussian Blending Engine (inside `utils_memory.py`)

For processing large images in overlapping tiles:
- **`generate_gaussian_window(window_size)`**: Creates a 2D Hanning window (`np.outer(np.hanning(n), np.hanning(n))`). This is the "confidence weight" — high in the center of a tile, tapering to zero at the edges.
- **`accumulate_blended_tile(...)`**: Adds each tile's SR output multiplied by its Gaussian window onto a master canvas. Also accumulates the weights separately.
- **Final stitch**: Divide master canvas by accumulated weights → seam-free output.

**Why Gaussian blending?** Without it, tile edges produce visible grid lines in the final image. The Gaussian taper ensures adjacent tiles blend smoothly at their overlapping regions.

### Step 5.4 — Build the Streamlit Evaluation UI

A `streamlit_app.py` with:
- File uploader for an LR `.tif` or `.png` image.
- Run inference button (triggers full 3-tier ensemble pipeline).
- Interactive side-by-side comparison slider (`streamlit_image_comparison`).
- Display panel showing PSNR, SSIM, and Hallucination Score.

### Step 5.5 — Final Inference on Delhi/Kanpur Test Scene

Run the full ensemble pipeline on a high-density urban area for the video demo deliverable.

### 📝 Journey Log Entry: "Stage 5 Complete: 3-Tier Hybrid Ensemble built. Key learning: Alpha map strategy — let physics/fidelity dominate in ambiguous terrain; let perception dominate in structured urban."
### 📖 Update README: Full project completion state with architecture diagram and final PSNR/SSIM table.
### 💾 Commit Checkpoint 5: "feat: stage 5 — 3-tier ensemble, Gaussian blending, Streamlit UI"

---

## Final Deliverable Checklist

| Item | Status |
|---|---|
| GitHub repo: clean code + `requirements.txt` + `README.md` | ⬜ |
| `PROJECT_JOURNEY.md`: full interview-ready log | ⬜ |
| Colab/Kaggle notebook for judge inference | ⬜ |
| 2-min video demo (Delhi/Kanpur scene, before/after slider) | ⬜ |
| PSNR/SSIM table vs. Bicubic baseline | ⬜ |
| Hallucination Score ≤ 0.05 on test set | ⬜ |

---

## Open Questions for Aditya to Confirm

> [!IMPORTANT]
> **Q1 — Upscale Factor**: Do you want to target **4x** (easier, more stable) or **8x** (harder, more impressive) as the primary deliverable? We can do both, but should lock in one for Stage 1 to keep the weight loading consistent.

> [!IMPORTANT]
> **Q2 — GEE Account**: Have you already registered at [https://earthengine.google.com](https://earthengine.google.com)? If not, do Step 0.3 now — approval can take up to 24 hours (though usually instant for students).

> [!NOTE]
> **Q3 — Local GPU?**: Do you have a local NVIDIA GPU with CUDA? If yes, Stage 1 inference can run locally. If no, we will run even Stage 1 on Kaggle.

> [!NOTE]
> **Q4 — Python Version**: Confirm your local Python version with `python --version`. We need `3.9+` for all dependencies to resolve cleanly.
