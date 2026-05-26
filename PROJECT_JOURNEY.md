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

---
<!-- Future entries will be appended here as we progress through each stage -->
