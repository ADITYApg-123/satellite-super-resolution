# 08 — Resume Defense

Every number and every claim on your resume will be challenged. This section prepares you to defend them with mathematical precision.

---

## Q1: You say 930K parameters. How did you calculate that?

### Why the interviewer is asking this
They want to know if you actually understand how parameter counts work or if you just ran `model.summary()`.

### Excellent Answer
> "I verified it by running `sum(p.numel() for p in model.parameters())` in PyTorch, which returned exactly 930,819.
>
> But I can also derive it manually. The architecture has:
>
> **conv_first**: 3 input channels × 64 output channels × 3×3 kernel + 64 bias = 1,792 parameters
>
> **8 Residual Blocks**, each containing:
> - conv1: 64×64×3×3 + 64 = 36,928
> - conv2: 64×64×3×3 + 64 = 36,928
> - SE block: 64→4 linear (256) + 4→64 linear (256) = 512
> - Total per block: ~74,368
> - 8 blocks: ~594,944
>
> **conv_after_body**: 64×64×3×3 + 64 = 36,928
>
> **Upsample (two stages)**: 2 × (64×256×3×3 + 256) = 2 × 147,712 = 295,424
>
> **conv_last**: 64×3×3×3 + 3 = 1,731
>
> Total ≈ 930K. The exact count matches what PyTorch reports."

### Common Mistakes
- Not being able to derive the count manually.
- Confusing parameters with FLOPs.

### Follow-up Questions
1. "How many FLOPs does a single forward pass require?"
2. "How does your parameter count compare to Real-ESRGAN?"
3. "Which layer has the most parameters? Why?"

---

## Q2: You claim 30.14 dB PSNR. How was this measured?

### Why the interviewer is asking this
They want to verify your evaluation methodology is rigorous.

### Excellent Answer
> "30.14 dB was the peak validation PSNR achieved at Epoch 49 during a 50-epoch training run on Kaggle. I can point you to the exact log line in my training log file.
>
> The measurement methodology: at the end of each epoch, the model switches to eval mode. I iterate through the validation set (393 images, the 10% hold-out split). For each image, the model produces a super-resolved output which is clamped to [0, 1]. I then compute PSNR between this output and the ground-truth HR image using the formula: PSNR = 10 × log10(1.0 / MSE), where MSE is the mean squared error across all RGB pixels.
>
> The PSNR is averaged across all 393 validation images to get the epoch's validation PSNR. I save the model weights only when this average exceeds the previous best, so `geosafe_best_generator.pth` corresponds to the 30.14 dB checkpoint.
>
> I should note that 30.14 dB was the peak — other epochs ranged from 29.07 to 30.14 dB, showing some variance. The model didn't plateau cleanly, suggesting there was still room for improvement with more epochs or learning rate adjustments."

### Common Mistakes
- Not knowing the PSNR formula.
- Reporting training PSNR instead of validation PSNR.
- Not mentioning the variance across epochs.

### Follow-up Questions
1. "Is 30.14 dB on the Y channel or RGB?"
2. "How does this compare to state-of-the-art on WorldStrat?"
3. "Why did Epoch 48 drop to 29.07 dB? What caused the variance?"

---

## Q3: You say "sub-second inference." Can you give the exact number?

### Why the interviewer is asking this
"Sub-second" is vague. They want precision.

### Excellent Answer
> "I should be transparent — 'sub-second' was based on qualitative observation during Streamlit testing, not a rigorous benchmark. For a production resume, I would run:
>
> ```python
> import time
> model.eval()
> input = torch.randn(1, 3, 64, 64)
> start = time.time()
> with torch.no_grad():
>     output = model(input)
> print(f'{(time.time() - start) * 1000:.1f} ms')
> ```
>
> For a more rigorous benchmark, I would average over 100 runs, exclude the first run (which includes CUDA kernel compilation), and test on multiple input sizes. The inference time depends heavily on whether we're running on CPU or GPU and the input dimensions.
>
> On a consumer CPU, I expect inference on a 64×64 patch to take approximately 50-150ms. On a T4 GPU, it would be under 10ms. The Streamlit app runs on CPU, and the 'sub-second' claim holds for typical input sizes."

### Common Mistakes
- Making up a specific number without having benchmarked it.
- Not acknowledging the dependency on input size and hardware.

### Follow-up Questions
1. "How does inference time scale with input resolution?"
2. "What is the theoretical peak throughput on a T4 GPU?"
3. "Would ONNX Runtime be faster than native PyTorch inference?"

---

## Q4: You claim the model was trained in 3.2 hours. Prove it.

### Why the interviewer is asking this
They might think 3.2 hours is suspiciously fast for a GAN.

### Excellent Answer
> "The training log file records timestamps. Epoch 1 started at approximately time 0, and Epoch 50 completed at 11,726.9 seconds. That's 11,726.9 / 3600 = 3.26 hours, which I rounded to 3.2 hours.
>
> This is fast for several reasons. First, the model is only 930K parameters — a tiny footprint compared to models like Real-ESRGAN (16M+). Second, the dataset is relatively small at 3,928 patches. Third, I used mixed-precision training on dual T4 GPUs, which roughly doubles the effective throughput. Fourth, the batch size of 4 with 64×64 patches is a very light workload for a T4.
>
> The processing speed was approximately 2.32 batches per second, which means ~3 minutes per epoch for 3,535 training samples. Over 50 epochs, that's 150 minutes of pure training time, plus validation overhead."

### Common Mistakes
- Not being able to trace the number back to the actual logs.
- Not explaining WHY it's fast (the interviewer might be skeptical).

### Follow-up Questions
1. "How much of the 3.2 hours was validation vs training?"
2. "If you doubled the dataset, would training time exactly double?"
3. "What was your GPU utilization? Was the GPU fully saturated?"

---

## Q5: You mention a "custom composite loss function." Many papers use multi-component losses. What's custom about yours?

### Why the interviewer is asking this
They're challenging whether your work is actually novel or just standard practice.

### Excellent Answer
> "You're right that multi-component losses are common in SR literature. What's custom about GeoSafeLoss is the specific combination and weighting hierarchy designed for the satellite domain.
>
> Most SR papers use L1 + VGG perceptual loss + GAN loss. My additions are:
>
> First, the Gradient Profile Loss using hardcoded Sobel filters. This is unusual — most papers rely on the perceptual loss to implicitly capture edges. I found that for satellite imagery, explicit edge supervision was necessary because the VGG-19 network (trained on ImageNet) doesn't have strong priors for the types of edges found in satellite data (building footprints, road networks, coastlines).
>
> Second, the weight hierarchy. I deliberately set the GAN weight to 0.005 — which is 5-10x lower than what most papers use. In natural image SR, mild hallucination is acceptable (nobody cares if the AI invents a slightly different grass texture). In satellite SR, any hallucination is a failure. The weight hierarchy reflects this domain-specific constraint.
>
> Third, the use of RaGAN specifically. Many papers still use vanilla GAN or WGAN-GP. I chose RaGAN because it provides more stable gradients at low weight, which is exactly what I needed — a GAN that contributes subtle texture improvement without dominating the loss landscape."

### Common Mistakes
- Claiming it's completely novel (it's not — it's a carefully tuned combination).
- Not being able to explain why each weight was chosen.

### Follow-up Questions
1. "Did you perform ablation studies? What happens if you remove one component?"
2. "How would you tune these weights for a different domain, like medical imaging?"
3. "What weight would you use if you didn't care about hallucination at all?"
