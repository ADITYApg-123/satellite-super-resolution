# 04 — Loss Engineering

This is the crown jewel of your project. If you nail these questions, you will stand out from 95% of candidates. Most junior engineers have never designed a custom loss function.

---

## Q1: Walk me through your GeoSafeLoss function. Why four components?

### Why the interviewer is asking this
They want to see if you understand the specific purpose of each loss term and why a single loss isn't enough.

### Excellent Answer
> "Each component of GeoSafeLoss addresses a specific failure mode in super-resolution:
>
> **L1 Loss (Weight: 1.0)** — This is the physics anchor. It computes the mean absolute pixel-wise difference between the predicted and ground-truth images. It ensures the model doesn't drift too far from mathematical correctness. I chose L1 over L2 (MSE) because L2 penalizes large errors quadratically, which causes the model to aggressively average pixels to avoid outliers — resulting in extreme blurriness. L1 is more tolerant of small edge misalignments.
>
> **Gradient Profile Loss (Weight: 0.30)** — Standard L1 treats all pixels equally, but edges are far more important in satellite imagery than flat regions. This loss applies Sobel edge-detection filters in both X and Y directions to extract edge maps from both the predicted and ground-truth images, then computes L1 between those edge maps. This forces the model to get edges razor-sharp even if the flat regions are slightly off.
>
> **LPIPS Loss (Weight: 0.04)** — LPIPS feeds both images through a frozen VGG-19 network and compares the intermediate feature activations. Unlike pixel-level metrics, this captures 'perceptual similarity' — whether the textures look right to a human eye. A field of grass might have slightly different pixel values but look identical to a human; LPIPS captures that. The weight is deliberately low (0.04) because we don't want perceptual quality to override geometric accuracy.
>
> **RaGAN Loss (Weight: 0.005)** — The Relativistic Average GAN loss provides the final layer of crispness. The Discriminator gives the Generator feedback on which regions of the image still look 'AI-generated.' The weight is extremely low (0.005) because GAN loss is the most dangerous component — if it's too strong, the Generator starts hallucinating fake structures to fool the Discriminator."

### Common Mistakes
- Listing the four losses without explaining WHY each weight is what it is.
- Not explaining the hierarchy: L1 is the base, everything else is a refinement.

### Follow-up Questions
1. "How did you choose those specific weights? Did you tune them?"
2. "What happens if you set the GAN weight to 0.1 instead of 0.005?"
3. "Could you add a fifth loss component? What would it be?"
4. "Why not SSIM loss? Many SR papers use it."
5. "What is the gradient of L1 loss at zero? Why does this matter?"

---

## Q2: What is LPIPS and how does it work internally?

### Why the interviewer is asking this
This tests whether you understand transfer learning and feature extraction with pretrained networks.

### Excellent Answer
> "LPIPS stands for Learned Perceptual Image Patch Similarity. It was developed by Richard Zhang at UC Berkeley.
>
> The key insight is that deep neural networks trained on ImageNet learn hierarchical feature representations that closely mirror human visual perception. Early layers detect edges and colors, middle layers detect textures and patterns, and deep layers detect semantic objects.
>
> LPIPS works like this: Take both the predicted image and the ground-truth image. Normalize them from [0,1] to [-1,1]. Feed both through a frozen VGG-19 network. Extract the feature activations at specific intermediate layers (typically conv1, conv2, conv3, conv4, conv5). For each layer, compute the L2 distance between the two sets of activations. Weight these distances by learned per-channel scaling factors (these weights were trained on a dataset of human perceptual judgments). Sum everything up.
>
> In my implementation, I wrap LPIPS in a custom `LPIPSWrapper` class. I freeze all VGG-19 parameters with `requires_grad=False` so backpropagation only flows through my Generator, not through VGG-19. The VGG-19 is purely a 'judge' — it never gets updated."

### Common Mistakes
- Saying "LPIPS uses a neural network to compare images" without explaining which layers or how.
- Not mentioning that VGG-19 is frozen during training.
- Confusing LPIPS with perceptual loss (they're related but LPIPS specifically uses learned weights calibrated on human judgments).

### Follow-up Questions
1. "Why VGG-19 and not ResNet-50 for LPIPS?"
2. "Which VGG-19 layers contribute most to the perceptual similarity score?"
3. "Does LPIPS add to your GPU memory usage? How much?"
4. "What is the difference between LPIPS and the 'perceptual loss' from Johnson et al.?"

---

## Q3: What is the Relativistic Average GAN (RaGAN) and why is it better than standard GAN loss?

### Why the interviewer is asking this
This is an advanced question. If you can answer this cleanly, you demonstrate deep GAN theory knowledge.

### Excellent Answer
> "In a standard GAN, the Discriminator outputs P(real) — the probability that an image is real. The Generator tries to maximize P(real) for fake images. The problem is that the Discriminator only sees one image at a time — it has no relative context.
>
> RaGAN changes the formulation. Instead of asking 'Is this image real?', it asks 'Is this real image MORE real than the average fake image?' The Discriminator now outputs a relative score: D(x_real) - E[D(x_fake)] for real images, and D(x_fake) - E[D(x_real)] for fake images.
>
> This has three benefits. First, the Generator gets signal even when the Discriminator is very confident, because the loss depends on the relative difference, not the absolute score. Second, it provides smoother gradients, which stabilizes training. Third, it forces both the Generator and Discriminator to improve symmetrically — the Generator wins by making fakes better, and the Discriminator wins by making its judgment more discriminative.
>
> The implementation uses BCEWithLogitsLoss on these relative predictions. For the Discriminator: it wants (real - mean(fake)) to be classified as 1, and (fake - mean(real)) to be classified as 0. For the Generator: the objectives are flipped."

### Common Mistakes
- Confusing RaGAN with WGAN or WGAN-GP (completely different approaches).
- Not explaining the 'averaging' component.
- Not knowing why standard GAN loss causes vanishing gradients.

### Follow-up Questions
1. "What is the vanishing gradient problem in standard GAN training?"
2. "How does WGAN-GP compare to RaGAN? Which is better and when?"
3. "What is mode collapse and could RaGAN prevent it?"

---

## Q4: Explain the Gradient Profile Loss. How do Sobel filters work?

### Why the interviewer is asking this
This tests classical computer vision knowledge alongside deep learning.

### Excellent Answer
> "The Gradient Profile Loss uses Sobel operators to extract edge information from both the predicted and ground-truth images.
>
> A Sobel filter is a 3×3 convolution kernel that approximates the first derivative of the image intensity. There are two kernels: one for horizontal edges (Gx) and one for vertical edges (Gy).
>
> Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  
> Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
>
> When convolved with a grayscale image, Gx produces high values where there's a sharp horizontal intensity change (like the edge of a building), and Gy does the same for vertical changes.
>
> In my implementation, I first convert the RGB image to grayscale by averaging across channels. Then I apply both Sobel kernels as fixed (non-learnable) 2D convolutions. I compute Sobel outputs for both the predicted and ground-truth images, and the loss is the L1 distance between their edge maps.
>
> The key design decision is that I register these kernels as `nn.Parameter` with `requires_grad=False`. This means they participate in the forward pass (so gradients flow through them to the Generator) but their values never change — they remain fixed as mathematical Sobel operators."

### Common Mistakes
- Not knowing the actual Sobel kernel values.
- Confusing Sobel with Canny edge detection.
- Not explaining why `requires_grad=False` is important.

### Follow-up Questions
1. "Why Sobel and not Canny or Laplacian?"
2. "Could you learn the edge detection kernels instead of using fixed Sobel filters?"
3. "What happens at the edges of the image? How do you handle boundary pixels?"

---

## Q5: How do you prevent the GAN from hallucinating fake structures?

### Why the interviewer is asking this
THIS is the question that defines your project. This is your star answer.

### Excellent Answer
> "Hallucination is the fundamental enemy of satellite image super-resolution. A GAN that's too creative will invent roads, buildings, or vegetation patterns that don't exist in reality. If a disaster response team uses such an image for decision-making, the consequences could be catastrophic.
>
> I prevent hallucination through a hierarchy of mathematical constraints:
>
> **Layer 1 — L1 Loss (Weight 1.0)**: This is the heaviest anchor. Every single pixel in the output must be as close as possible to the ground truth. This prevents the model from 'drifting' into fantasy.
>
> **Layer 2 — Gradient Profile Loss (Weight 0.30)**: Even if individual pixels are close, the model could still create false edges. The Sobel filter explicitly checks that every edge in the output corresponds to a real edge in the ground truth. A hallucinated building would create edges that don't exist in the reference edge map, triggering a massive penalty.
>
> **Layer 3 — Extremely Low GAN Weight (0.005)**: The GAN is kept on a very tight leash. Its contribution to the total loss is 200x smaller than L1. It can only 'suggest' subtle textural improvements, not override the geometric constraints.
>
> **Layer 4 — Validation PSNR Monitoring**: At the end of every epoch, I compute PSNR on a held-out validation set. If hallucination were occurring, PSNR would drop because the hallucinated details don't match the ground truth. I save the model only when PSNR improves.
>
> The result is that the GAN adds crispness and texture without the freedom to invent structures."

### Common Mistakes
- Saying "I used L1 loss to prevent hallucination" without explaining the full hierarchy.
- Not mentioning the weight ratios (the actual numbers are what make this convincing).

### Follow-up Questions
1. "How would you implement Cycle Consistency Loss as an additional anti-hallucination measure?"
2. "Could you quantify hallucination? Is there a 'hallucination score'?"
3. "What if the model hallucinates in a way that happens to improve PSNR?"
