# 03 — ML and Deep Learning

These questions test your fundamental understanding of how neural networks learn. Expect these at any ML-focused company.

---

## Q1: How did you train the model? Walk me through the training loop.

### Why the interviewer is asking this
They want to know if you actually wrote the training loop or just called `model.fit()`. GAN training is notoriously tricky.

### Excellent Answer
> "This is a GAN-based training loop, so it alternates between training two separate networks: the Generator (which produces super-resolved images) and the Discriminator (which judges whether an image is real or AI-generated).
>
> **Step A — Train the Discriminator**: I freeze the Generator and generate a batch of fake HR images. I feed both the real HR images and the fake HR images to the Discriminator. It outputs a per-pixel 'realness' map for each. The Discriminator's loss pushes it to correctly identify real images as real and fakes as fake. Crucially, I call `.detach()` on the fake images before feeding them to the Discriminator — this prevents gradients from accidentally flowing back into the Generator during this step, which would cause GPU memory to explode.
>
> **Step B — Train the Generator**: Now I freeze the Discriminator. The Generator produces another batch of fake HR images (we need fresh gradients). The GeoSafeLoss computes four components: L1 for pixel accuracy, Gradient Profile for edge sharpness, LPIPS for perceptual texture quality, and RaGAN for adversarial crispness. The total loss backpropagates through the Generator only.
>
> **Step C — Validation**: At the end of each epoch, I switch to `model.eval()`, disable gradients with `torch.no_grad()`, and compute PSNR on a held-out 10% validation set.
>
> I also use mixed-precision training with `torch.cuda.amp.autocast()` and `GradScaler` to cut GPU memory usage nearly in half and speed up training."

### Common Mistakes
- Not mentioning the `.detach()` call (this is a critical GAN training detail).
- Not knowing the difference between `model.train()` and `model.eval()`.
- Forgetting to mention mixed-precision training.

### Follow-up Questions
1. "What happens if you don't detach the fake images in the discriminator step?"
2. "Why do you generate fake images twice — once for D training and once for G training?"
3. "What is mixed-precision training and why does it save memory?"
4. "What's the difference between `torch.no_grad()` and `model.eval()`?"
5. "How do you handle mode collapse in GAN training?"

---

## Q2: What optimizer did you use and why?

### Why the interviewer is asking this
Optimizer choice reveals how deeply you understand training dynamics.

### Excellent Answer
> "I used AdamW for both the Generator and Discriminator. AdamW is Adam with decoupled weight decay regularization. 
>
> Standard Adam applies weight decay as part of the gradient update, which means the decay interacts with the adaptive learning rate in unexpected ways. AdamW fixes this by applying weight decay directly to the weights after the gradient step, which has been shown to produce better generalization.
>
> I set the learning rate to 2e-4 with a weight decay of 1e-4. For the learning rate schedule, I used CosineAnnealingWarmRestarts with T_0=10 and T_mult=2. This means the learning rate follows a cosine curve that decays to 1e-6 over the first 10 epochs, then 'restarts' with a warm learning rate. The T_mult=2 means each subsequent cycle is twice as long (10 epochs, then 20, then 40). This helps the model escape local minima by periodically boosting the learning rate."

### Common Mistakes
- Saying "I used Adam because everyone uses Adam."
- Not knowing the difference between Adam and AdamW.
- Not being able to explain the learning rate scheduler.

### Follow-up Questions
1. "Why not SGD with momentum? When would SGD outperform Adam?"
2. "What is the mathematical formula for the Adam update rule?"
3. "What does T_mult=2 do in CosineAnnealingWarmRestarts?"
4. "How did you choose the initial learning rate of 2e-4?"

---

## Q3: What is mixed-precision training and why did you use it?

### Why the interviewer is asking this
This tests practical GPU engineering knowledge — something that separates junior from senior ML engineers.

### Excellent Answer
> "Mixed-precision training uses both FP16 (16-bit floating point) and FP32 (32-bit floating point) during training to reduce memory usage and speed up computation.
>
> Modern GPUs like the T4 have specialized Tensor Cores that can perform FP16 matrix multiplications 2-8x faster than FP32. By wrapping the forward pass in `torch.cuda.amp.autocast()`, PyTorch automatically casts operations to FP16 where it's safe (like convolutions and matrix multiplies) while keeping operations that need precision in FP32 (like loss computation and batch normalization).
>
> The tricky part is that FP16 has a much smaller numerical range. Gradients can underflow to zero during backpropagation. The `GradScaler` solves this by multiplying the loss by a large scale factor before backpropagation (so gradients stay in FP16's representable range), and then dividing the gradients back down before the optimizer step. If the scaler detects NaN or Inf gradients, it skips that optimizer step and reduces the scale factor.
>
> In my case, this reduced GPU memory usage from ~14GB to ~8GB on the T4s, which was critical because the T4 only has 16GB of VRAM."

### Common Mistakes
- Saying "it makes training faster" without explaining the mechanism.
- Not knowing what GradScaler does or why it's necessary.

### Follow-up Questions
1. "What operations are kept in FP32 and why?"
2. "What happens if the GradScaler detects NaN gradients?"
3. "What is BF16 and how does it differ from FP16?"

---

## Q4: How did you handle overfitting?

### Why the interviewer is asking this
With only 3,928 training samples and 930K parameters, overfitting is a legitimate concern.

### Excellent Answer
> "I used several strategies. First, data augmentation — random 90/180/270 degree rotations and horizontal/vertical flips. Since satellite imagery has no preferred orientation (unlike face photos), these augmentations are semantically valid and effectively multiply the dataset size by 8x.
>
> Second, the architecture itself provides implicit regularization. The 64-channel bottleneck and the global residual connection constrain the model's capacity. The SE blocks with a reduction ratio of 16 also act as a bottleneck, compressing 64 channels to 4 before re-expanding.
>
> Third, AdamW's decoupled weight decay (1e-4) provides L2 regularization.
>
> Fourth, I used a 90/10 train-validation split and tracked validation PSNR every epoch. I saved the model checkpoint only when validation PSNR improved. If validation PSNR started decreasing while training loss continued to drop, that would indicate overfitting — though in practice, the model reached convergence without significant overfitting in 50 epochs.
>
> Fifth, gradient clipping with `max_norm=1.0` prevents gradient explosion, which indirectly helps with overfitting by preventing the model from making overly aggressive weight updates."

### Common Mistakes
- Only mentioning one technique (like dropout) and not having a holistic strategy.
- Not knowing that data augmentation for satellite images is different from natural images (you can't do color jittering aggressively because colors have physical meaning in multispectral data).

### Follow-up Questions
1. "Why didn't you use Dropout?"
2. "At what epoch did overfitting start to appear?"
3. "Could you use more aggressive augmentations like CutMix or MixUp for satellite data?"

---

## Q5: What is gradient clipping and why did you use it?

### Why the interviewer is asking this
GAN training is notoriously unstable. This tests if you know why.

### Excellent Answer
> "Gradient clipping caps the magnitude of gradients during backpropagation. I used `clip_grad_norm_` with `max_norm=1.0`, which scales down the entire gradient vector if its L2 norm exceeds 1.0, while preserving the direction.
>
> This is critical for GAN training because the adversarial loss can produce wildly large gradients. If the Discriminator becomes too confident, it sends extreme gradients to the Generator, which can cause the Generator's weights to change dramatically in a single step — destabilizing training or causing NaN values. Gradient clipping acts as a safety valve that prevents these catastrophic updates.
>
> I applied clipping only to the Generator, not the Discriminator, because the Generator is the one receiving the adversarial signal and is more vulnerable to gradient explosions."

### Common Mistakes
- Confusing gradient clipping with gradient penalty (they are different concepts).
- Not knowing the difference between `clip_grad_norm_` and `clip_grad_value_`.

### Follow-up Questions
1. "What's the difference between clipping by norm vs clipping by value?"
2. "How did you choose max_norm=1.0?"
3. "What are other techniques to stabilize GAN training?"

---

## Q6: What is PSNR and why is it your primary metric?

### Why the interviewer is asking this
They want to see if you understand the limitations of your chosen metric.

### Excellent Answer
> "PSNR stands for Peak Signal-to-Noise Ratio. It measures the ratio between the maximum possible pixel intensity and the mean squared error between the predicted and ground-truth images, expressed in decibels. The formula is: PSNR = 10 × log10(MAX² / MSE).
>
> Higher PSNR means the predicted image is closer to the ground truth. In super-resolution, anything above 28 dB is considered good, and above 30 dB is considered excellent. My model achieves 30.14 dB.
>
> However, PSNR has a known limitation — it doesn't perfectly correlate with human visual perception. A blurry image that's mathematically 'safe' can score higher PSNR than a sharp image with slightly misaligned edges. That's precisely why I also use LPIPS (a perceptual metric) in my loss function. PSNR ensures mathematical correctness; LPIPS ensures visual quality."

### Common Mistakes
- Not knowing the formula.
- Not mentioning PSNR's limitations (this is what separates a strong candidate).
- Confusing PSNR with SNR.

### Follow-up Questions
1. "What's the relationship between PSNR and MSE?"
2. "What is SSIM and how does it differ from PSNR?"
3. "A model gets 32 dB PSNR but the images look blurry. Another gets 29 dB but looks sharp. Which is better?"
4. "What is the theoretical maximum PSNR?"

---

## Q7: Your model was trained on Kaggle. How did you handle GPU timeouts?

### Why the interviewer is asking this
This tests practical MLOps experience. Cloud GPU management is a real-world skill.

### Excellent Answer
> "Kaggle kills GPU sessions after approximately 12 hours or if you exceed the weekly quota. This means multi-day training runs are impossible without a robust checkpoint system.
>
> I implemented a comprehensive checkpoint-resume mechanism that saves not just the model weights, but the complete training state: Generator weights, Discriminator weights, both AdamW optimizer states (including momentum buffers and second-moment estimates), both learning rate scheduler states, the current epoch number, and the best PSNR achieved so far.
>
> When the notebook restarts, the `load_checkpoint` function restores everything exactly, so training continues seamlessly. The critical detail that most people miss is saving the optimizer state. If you only save model weights and reinitialize the optimizer from scratch, the momentum buffers reset to zero. The first few epochs after resume then see a massive destructive spike in the loss because Adam's adaptive learning rates are suddenly based on zero history. By saving the full optimizer state, the loss curve is perfectly smooth across restarts."

### Common Mistakes
- Only saving model weights without optimizer state.
- Not knowing why optimizer state matters for training continuity.

### Follow-up Questions
1. "How large is each checkpoint file?"
2. "What if the checkpoint gets corrupted? Do you have a backup strategy?"
3. "How would you implement distributed training across multiple machines?"
