# 06 — Debugging and Technical Challenges

These questions test your problem-solving ability under pressure. Interviewers want to hear real war stories, not theoretical knowledge.

---

## Q1: What was the hardest bug you encountered?

### Why the interviewer is asking this
Debugging ability is the #1 predictor of engineering success. They want a real story with a real resolution.

### Excellent Answer
> "The hardest bug was the 12-channel C++ crash. When I pushed my entire pipeline to Kaggle and started the first epoch, OpenCV's C++ backend immediately crashed with: 'Unsupported number of channels: channels >= 1 && channels <= 4 where channels is 12.'
>
> This was hard for three reasons. First, the error was in C++ code inside OpenCV's TiffDecoder, not in my Python code, so the stack trace was nearly unreadable. Second, it only appeared on Kaggle — locally I was testing with PNG samples that happened to be 3-channel, so I never hit this code path. Third, the error flooded the console with hundreds of identical warning lines, making it impossible to find the actual root cause.
>
> My debugging process: I first ran a diagnostic cell that used `rasterio` to inspect the raw file metadata. I discovered the Sentinel-2 GeoTIFFs had 12 spectral bands. I then checked OpenCV's source code and confirmed it has a hardcoded limit of 4 channels. The fix was to completely remove OpenCV from the data loading pipeline and replace it with rasterio, which handles arbitrary-channel rasters natively. I then wrote explicit band-slicing logic to extract RGB bands [3, 2, 1].
>
> This bug taught me that you should always validate your data loading pipeline with the actual production data format before writing any model code."

### Common Mistakes
- Telling a story where the bug was trivial (like a typo).
- Not explaining the systematic debugging process.

### Follow-up Questions
1. "How long did it take you to diagnose this?"
2. "How would you write a test to prevent this from happening again?"
3. "What other data formats could cause similar issues?"

---

## Q2: Tell me about the checkpoint resumption bug.

### Why the interviewer is asking this
This tests MLOps maturity. Most junior engineers make this exact mistake.

### Excellent Answer
> "When I resumed training after a Kaggle timeout, the loss curve showed a massive destructive spike in the first 2-3 epochs after resumption. The model's performance temporarily degraded before slowly recovering.
>
> I diagnosed it by comparing two scenarios: a fresh 50-epoch training run versus a run that was interrupted at epoch 25 and resumed. The resumed run had a clear spike at epoch 26. I hypothesized that the issue was optimizer state loss.
>
> AdamW maintains two internal buffers for each parameter: the first moment (mean of recent gradients) and the second moment (mean of recent squared gradients). These buffers encode the 'training history' and are used to compute adaptive per-parameter learning rates. When I was only saving `model.state_dict()` and reinitializing the optimizer from scratch, these buffers reset to zero. The optimizer suddenly had no history, so it computed wildly inappropriate learning rates for the first several steps.
>
> The fix was comprehensive: I modified `save_checkpoint()` to serialize the Generator state, Discriminator state, Generator optimizer state, Discriminator optimizer state, both learning rate scheduler states, the current epoch, and the best PSNR. On resume, everything is restored exactly. The loss curve became perfectly smooth across restarts."

### Common Mistakes
- Not understanding why optimizer state matters.
- Saying "I just saved the model and loaded it back" without mentioning the spike.

### Follow-up Questions
1. "How large is a full checkpoint compared to just model weights?"
2. "What if you're resuming with a different batch size? Does that break anything?"
3. "How does learning rate warmup relate to this problem?"

---

## Q3: How did you debug the DataLoader finding zero matches?

### Why the interviewer is asking this
Data pipeline debugging is the most common real-world ML engineering challenge.

### Excellent Answer
> "When I first ran the training script on Kaggle, the DataLoader reported 0 paired images found. The WorldStrat dataset was clearly there (I could see the files), but the matching logic was failing.
>
> I wrote a diagnostic cell that recursively walked the file system using `os.walk()` and printed the complete directory tree. I discovered that the high-resolution Airbus images were not at the path I expected. They were hidden inside an unlisted `hr_dataset/12bit/` subfolder, and each location contained four different file variants (panchromatic, pansharpened, RGB, raw).
>
> My DataLoader was trying to match filenames directly between the LR and HR directories, but the naming conventions were completely different. The fix was threefold: first, I updated the HR path to point to the `12bit/` subdirectory. Second, I added a filename filter to specifically target files ending in `_ps.tiff` (the pansharpened variant). Third, I implemented a matching function that used the location ID (embedded in the folder name) rather than exact filename matching.
>
> After these fixes, the DataLoader successfully found and paired all 3,928 locations."

### Common Mistakes
- Assuming the dataset structure matches the documentation.
- Not writing diagnostic scripts to inspect the actual file system.

### Follow-up Questions
1. "How would you make this DataLoader more robust for future datasets?"
2. "What if some locations had LR images but no HR images?"
3. "How did you verify that the LR-HR pairs were actually aligned?"

---

## Q4: Did you encounter any GPU memory issues?

### Why the interviewer is asking this
VRAM management is a critical practical skill for ML engineers.

### Excellent Answer
> "Yes. The T4 GPU on Kaggle has only 16GB of VRAM. Initially, with a batch size of 4 and full FP32 precision, I was hitting OOM (Out of Memory) errors.
>
> I solved this with three techniques. First, mixed-precision training using `torch.cuda.amp.autocast()` and `GradScaler`, which reduced memory usage by roughly 40% by computing most operations in FP16. Second, I carefully managed the computational graph by calling `.detach()` on fake images before feeding them to the Discriminator during its training step. Without `.detach()`, PyTorch would retain the entire Generator's computational graph in memory while training the Discriminator, effectively doubling VRAM usage. Third, I set `pin_memory=True` in the DataLoader, which uses page-locked CPU memory for faster CPU-to-GPU transfers, reducing the time that both CPU and GPU copies of the data coexist in memory.
>
> In the Streamlit app, I added a different kind of memory protection — dynamic input size caps. If the user selects the 4x model, inputs are capped at 512px. If they select the 8x model, inputs are capped at 256px. This prevents the app from crashing on consumer hardware."

### Common Mistakes
- Not mentioning `.detach()` as a memory optimization (it's critical for GANs).
- Not knowing what `pin_memory` does.

### Follow-up Questions
1. "What is the exact VRAM breakdown? How much does each component use?"
2. "What other techniques exist for reducing VRAM usage?"
3. "What is gradient accumulation and when would you use it?"
4. "What is the difference between `del tensor` and `tensor = None` for freeing GPU memory?"

---

## Q5: If you had to debug this project from scratch with no logs, where would you start?

### Why the interviewer is asking this
This tests systematic debugging methodology — a senior engineering skill.

### Excellent Answer
> "I would follow a strict bottom-up debugging strategy:
>
> **Step 1 — Data Sanity Check**: Before touching any model code, I would load a single LR-HR pair and visually inspect it. Are the images aligned? Are the pixel values in the expected range? Are the dimensions correct? I would plot histograms of pixel values to check for normalization issues.
>
> **Step 2 — Forward Pass Check**: I would pass a single batch through the Generator with random weights and verify the output shape. For a 64x64 input with 4x upscaling, the output should be 256x256x3. If the shape is wrong, the architecture has a bug.
>
> **Step 3 — Loss Sanity Check**: I would compute each loss component individually and verify they're all returning reasonable positive values. A loss of 0 or NaN immediately indicates a bug.
>
> **Step 4 — Overfit on One Sample**: I would train the model on a single image pair for 100 epochs. If the model can perfectly reconstruct one image (PSNR > 40 dB), the architecture and training loop are working. If it can't overfit one sample, something is fundamentally broken.
>
> **Step 5 — Gradual Scale-Up**: Once the single-sample test passes, I would train on 10 samples, then 100, then the full dataset, checking for unexpected behavior at each scale."

### Common Mistakes
- Starting debugging at the model level without checking the data first.
- Not mentioning the single-sample overfitting test (this is a gold-standard technique).

### Follow-up Questions
1. "How would you debug if PSNR plateaus at 25 dB and won't improve?"
2. "What if the loss is NaN? What are the possible causes?"
3. "How would you diagnose mode collapse in the GAN?"
