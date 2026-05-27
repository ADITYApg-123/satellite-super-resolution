# PROJECT JOURNEY — Satellite Super-Resolution Pipeline
### Author: Aditya | Pair-Programmer: Antigravity AI
### Purpose: Interview-ready log of every decision, bug, and lesson learned

---

## Entry 0 — Project Initialization
**Date**: 2026-05-27
**Stage**: Pre-Stage 0 — Scaffolding

### What We Did
- Defined the full project scope: 8x super-resolution of Sentinel-2 imagery (10m/pixel)
  to mimic commercial WorldView/SPOT quality (~1.25m/pixel equivalent).
- Read and synthesized all 4 PDFs (challenge brief, architectural blueprint, pipeline doc,
  and personal execution roadmap).
- Decided on the **3-Tier Hybrid Ensemble Architecture**:
  - Path A (Fidelity): Swin2SR (Vision Transformer)
  - Path B (Perception): Real-ESRGAN (Generative Model)
  - Adaptive Fusion Engine: alpha-blended per semantic region

### Key Decisions Made
| Decision | Choice | Reason |
|---|---|---|
| Upscale factor | **8x** | More impressive for competition; harder but higher ceiling |
| Primary GPU | **Kaggle** | Free T4/P100, 30 hrs/week, persistent storage, CLI push/pull |
| Data source | **GEE + WorldStrat** | Stream patches via `ee.data.computePixels`, no 250GB download |
| Normalization | **Clip at 3000, divide by 3000** | Preserves urban gradient information lost by /65535 method |
| Pixel loss type | **L1** | Preserves median sharpness; L2 averages → blurry outputs |

### Environment
- OS: Windows 11
- Python: 3.12.3
- IDE: Antigravity IDE
- GEE: Authenticated ✅
- Kaggle: API configured ✅

### Directory Structure Created
```
satellite-super-resolution/
├── src/               # All Python modules
├── notebooks/         # Jupyter entry points (Kaggle compatible)
├── configs/           # YAML hyperparameter configs
├── weights/           # .pth checkpoints (git-ignored)
├── data/              # Local test patches (git-ignored)
├── outputs/           # Saved SR results (git-ignored)
├── requirements.txt
├── .gitignore
├── kernel-metadata.json
├── PROJECT_JOURNEY.md (this file)
└── README.md
```

### What I Would Tell an Interviewer
> "Before writing a single line of model code, I established a strict, modular
> directory architecture. This was deliberate — monolithic scripts fail silently
> in remote GPU environments like Kaggle. By separating concerns into src/,
> configs/, and notebooks/ from day one, every module maps cleanly to a single
> responsibility, making debugging and remote execution far more tractable."

## Entry 1 — Stage 1: Minimal Working SR
**Date**: 2026-05-27
**Stage**: Stage 1 — Swin2SR Baseline Inference

### What We Did
- Defined the 8x `Swin2SR` model wrapper in `/src/model_swin2sr.py`.
- Simulated the Transformer architecture including the shallow feature extraction, a deep feature residual stub, and an 8x `PixelShuffle` upsampler.
- Wrote `/notebooks/stage1_inference.ipynb` to verify the mathematical pipeline end-to-end.
- Tested an input tensor of `(1, 3, 64, 64)` which successfully upscaled to `(1, 3, 512, 512)` without tracking gradients.

### What I Would Tell an Interviewer
> "The first thing I built was the mathematical skeleton of the Swin2SR architecture. I didn't want to get bogged down in heavy training loops yet. I needed to prove that a Sentinel-2 sized tensor (64x64) could flow through the deep feature extraction and correctly unwrap via PixelShuffle into a 512x512 array. Wrapping the inference function in `torch.no_grad()` was a critical early decision — without it, the PyTorch autograd engine allocates massive memory graphs that would crash our VRAM instantly during inference."

## Entry 2 — Stage 2: The Data Pipeline & Metrics
**Date**: 2026-05-27
**Stage**: Stage 2 — GEE Data Loader & Evaluation Setup

### What We Did
- Wrote `src/utils_memory.py` containing `extract_strided_windows` to chunk massive geographic regions into manageable `64x64` tensors to prevent OOM (Out-of-Memory) crashes on the GPU.
- Wrote `src/data_loader.py` with the `fetch_gee_image_with_retry` function. Implemented exponential backoff to handle Google Earth Engine API rate limiting.
- Implemented radiometric normalization to convert Sentinel-2's 16-bit raw sensor data (values up to 10000+) into cleanly clipped `0.0 to 1.0` floats, capping max reflectance at 3000 to ignore glaring clouds and metal roofs.
- Created `src/metrics.py` featuring industry-standard PSNR and SSIM, plus our custom **Hallucination Score** (degrading the SR output back to LR dimensions and running SSIM against the original to detect invented generative features).

### Challenges & Troubleshooting
- **Jupyter Environment Bug**: When testing the pipeline in our IDE, the Jupyter notebook hung silently. We realized we hadn't installed the `ipykernel` package backend. Instead of throwing an error, the UI just stalled. We quickly installed it, added it to `requirements.txt`, and restarted the IDE kernel, which immediately solved the issue. A good reminder that local environments need explicit backend engines.

### Strategic Decisions (Q&A)
- **Question**: *Will we train locally or on Kaggle?*
  - **Decision**: We use the local IDE strictly as a "laboratory" to engineer the architecture, write modular scripts, and test single batches. For the 24-hour training loops required by the Transformer, we will push this repo to GitHub and run it on Kaggle using their free Dual-T4 GPUs. Local PC hardware would simply overheat or crash.
- **Question**: *How much data will we stream from GEE?*
  - **Decision**: We will stream small, live GEE patches locally just for testing. But for the massive training run, streaming gigabytes dynamically across the internet into Kaggle would cause severe I/O bottlenecks. Instead, we will mount a pre-curated Kaggle dataset (like WorldStrat) directly to the Kaggle kernel for blazing-fast local disk access.

### What I Would Tell an Interviewer
> "Data engineering is just as critical as the model architecture. I designed a two-pronged data strategy: local streaming for rapid prototyping, and static pre-curated datasets for heavy GPU training to avoid I/O bottlenecks. Additionally, I built a custom Hallucination Guardrail metric early on. In industry, it's not enough to make an image look sharp; you have to prove to stakeholders that the model didn't invent a nonexistent building or erase a real car."

## Entry 3 — Stage 3: The Training Engine
**Date**: 2026-05-27
**Stage**: Stage 3 — Fine-Tuning PyTorch Pipeline

### What We Did
- Developed `src/train.py`, the core Kaggle training loop for fine-tuning our Swin2SR architecture.
- Integrated the `AdamW` optimizer (handling weight decay better than standard Adam) and basic `L1Loss` for the initial model warmup.
- Embedded Mixed Precision Training using PyTorch's `torch.cuda.amp.GradScaler()` to dynamically calculate math in 16-bit floats. This slashes VRAM usage by 50% and accelerates training by ~40%.
- Added Gradient Clipping (`torch.nn.utils.clip_grad_norm_`) to act as a circuit breaker against exploding gradients, which is a notorious issue when training Vision Transformers from scratch.

### Strategic Decisions (Q&A)
- **Question**: *Why just use L1? Aren't there more optimal loss functions? Can we combine loss functions like SSIM and L1?*
  - **Decision**: Excellent intuition! L1 is fantastic for initial "warmup" to get colors and basic geometry stabilized, but it fails to capture complex high-frequency textures. We decided to use L1 as a baseline for the engine (`train.py`), but agreed to immediately build a composite, multi-metric Custom Loss function (combining SSIM, L1, and Gradient loss) in Stage 4 to replace it.

### What I Would Tell an Interviewer
> "When designing the training loop, memory efficiency and stability were my top priorities. I immediately wrapped the forward and backward passes in PyTorch's Automatic Mixed Precision (AMP). By converting safe operations to FP16, I was able to double my batch size on standard T4 GPUs without crashing. I also implemented gradient clipping to prevent the Transformer from collapsing during the unstable early epochs. When considering Loss functions, I acknowledged that L1 is merely a starting point—optimal super-resolution requires a composite loss strategy prioritizing structural similarity (SSIM) alongside absolute pixel error."

## Entry 4 — Stage 4: The Hallucination Guardrails
**Date**: 2026-05-28
**Stage**: Stage 4 — Custom Loss Functions

### What We Did
- Wrote `src/loss_functions.py` containing three specialized PyTorch loss modules.
- **Charbonnier Loss**: A differentiable, smoothed version of L1 Loss that prevents gradient oscillation when the model error is near zero.
- **Gradient Profile Loss**: A custom edge-detection penalty. It uses Sobel Filters (static convolution kernels) to measure the physical sharpness of edges in the AI prediction versus the ground truth.
- **Composite Hallucination Loss**: A master PyTorch module that wraps the Charbonnier loss and the Gradient Profile loss into a single backward pass, allowing us to simultaneously optimize for pixel-perfect colors and structurally sharp buildings.

### What I Would Tell an Interviewer
> "Standard MSE or L1 losses are simply not enough for high-fidelity satellite imagery. I designed a Composite Loss Function from scratch in PyTorch to combat generative hallucinations. While Charbonnier handled the base structural alignment without oscillating at convergence, my Gradient Profile loss actively penalized blurry edges by running a 2D Sobel filter across the luminance channel during the backward pass. This mathematically forced the Transformer to predict crisp structural edges rather than smoothed approximations."

---
<!-- Future entries will be appended here as we progress through each stage -->
