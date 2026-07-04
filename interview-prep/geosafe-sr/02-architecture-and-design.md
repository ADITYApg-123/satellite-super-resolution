# 02 — Architecture and Design

These questions drill into the exact structure of your neural network. Interviewers at ML-focused companies will spend the most time here.

---

## Q1: Walk me through the architecture of your model.

### Why the interviewer is asking this
They want to see if you can draw the model on a whiteboard. If you can't explain the data flow from input to output, they'll assume you copied code without understanding it.

### Excellent Answer
> "The architecture is a Deep Residual Attention Network with 4x upscaling. Let me walk through it layer by layer.
>
> **Stage 1 — Shallow Feature Extraction**: A single 3x3 convolution takes the 3-channel RGB input and expands it to 64 feature channels. This captures basic low-level features like edges and color gradients.
>
> **Stage 2 — Deep Feature Extraction**: 8 sequential Residual Blocks process these 64 channels. Each block contains two 3x3 convolutions with a ReLU activation in between, plus a Squeeze-and-Excitation channel attention module. A skip connection adds the block's input to its output, so the network only needs to learn the 'residual' difference — this makes training much more stable.
>
> **Stage 3 — Global Residual Connection**: After all 8 blocks, the output passes through one more 3x3 conv, and then the original shallow features from Stage 1 are added back. This is a second level of skip connection that prevents gradient vanishing across the entire depth of the network.
>
> **Stage 4 — Upsampling**: We do 4x upscaling in two steps of 2x each. Each step uses a 3x3 conv to expand channels from 64 to 256, followed by PixelShuffle(2) which rearranges those 256 channels into a spatially larger image with 64 channels. Two rounds of this gives us 4x spatial resolution.
>
> **Stage 5 — Reconstruction**: A final 3x3 conv maps the 64 feature channels back to 3 RGB channels, producing the super-resolved output."

### Common Mistakes
- Saying "I used a CNN" without explaining the specific blocks.
- Not mentioning the skip connections (they are critical to why the model works).
- Confusing PixelShuffle with transpose convolution.

### Follow-up Questions
1. "Why 8 residual blocks? Why not 4 or 16?"
2. "Why PixelShuffle instead of transposed convolution for upsampling?"
3. "What happens if you remove the global skip connection?"
4. "How does the receptive field change with 8 blocks?"
5. "Why 64 feature channels? What happens if you use 128?"

---

## Q2: What is a Residual Block and why did you use it?

### Why the interviewer is asking this
This tests foundational deep learning knowledge. Every ML engineer should know ResNets inside-out.

### Excellent Answer
> "A Residual Block learns the difference (residual) between the input and the desired output, rather than learning the full transformation from scratch. Mathematically, instead of learning F(x), the network learns F(x) = H(x) - x, and the output is x + F(x).
>
> This solves two critical problems. First, the vanishing gradient problem — in very deep networks, gradients shrink to near-zero as they backpropagate through many layers, making the early layers unable to learn. The skip connection provides a 'gradient highway' that lets gradients flow directly to earlier layers. Second, it makes the optimization landscape smoother — the network can always fall back to the identity function (just passing the input through) if the block can't learn anything useful, which prevents degradation.
>
> I used residual blocks because super-resolution is fundamentally a residual problem. The low-res and high-res images share 90% of the same information. The model only needs to learn the 'missing detail' — the texture, the sharpness, the fine edges. Residual learning is perfectly suited for this."

### Common Mistakes
- Only saying "it helps with deep networks" without explaining the gradient mechanism.
- Not connecting residual learning to the super-resolution task specifically.

### Follow-up Questions
1. "What's the difference between a pre-activation and post-activation ResBlock?"
2. "Could you use DenseNet-style connections instead? What would change?"
3. "What is the mathematical proof that skip connections help gradient flow?"

---

## Q3: What is Squeeze-and-Excitation (SE) Channel Attention?

### Why the interviewer is asking this
This tests whether you understand attention mechanisms beyond just Transformers. SE blocks are elegant and widely used.

### Excellent Answer
> "Squeeze-and-Excitation is a channel attention mechanism that allows the network to dynamically learn which feature channels are most important for a given input.
>
> **Squeeze**: We apply Global Average Pooling to compress each of the 64 feature channels into a single number. So a 64×H×W tensor becomes a 64×1×1 vector. This 'squeezes' the spatial information into a channel descriptor.
>
> **Excitation**: This 64-dimensional vector passes through a small bottleneck MLP — first a linear layer that compresses it from 64 to 4 (reduction ratio of 16), then ReLU, then another linear layer that expands it back to 64, then Sigmoid. The output is 64 values between 0 and 1.
>
> **Scale**: We multiply each original feature channel by its corresponding excitation weight. If the network decides that 'edge features' are important for this particular patch (say, a city scene), it will upweight those channels. If it's processing a water body, it might upweight 'texture smoothness' channels instead.
>
> The beauty is that this adds almost zero parameters — just two small linear layers — but gives the network adaptive, input-dependent behavior. It's like giving the CNN a tiny attention mechanism without the quadratic cost of full self-attention."

### Common Mistakes
- Confusing channel attention with spatial attention.
- Not knowing the reduction ratio or why it exists (it's a parameter efficiency tradeoff).
- Saying "it's like Transformers" without explaining the actual mechanism.

### Follow-up Questions
1. "What's the parameter cost of adding SE to each of your 8 blocks?"
2. "Why a reduction ratio of 16? What if you used 4 or 32?"
3. "How does SE compare to CBAM (Convolutional Block Attention Module)?"
4. "Could you replace SE blocks with full self-attention? What would the tradeoff be?"

---

## Q4: Why PixelShuffle for upsampling instead of transposed convolution?

### Why the interviewer is asking this
This is a classic deep learning interview question. The choice of upsampling method directly affects output quality.

### Excellent Answer
> "Transposed convolution (also called deconvolution) is the naive approach to upsampling, but it has a well-documented problem called 'checkerboard artifacts.' Because of how the kernel slides with stride > 1, certain output pixels receive contributions from more kernel positions than others, creating a grid-like pattern in the output. This is devastating for satellite imagery where we need clean, uniform textures.
>
> PixelShuffle (also called sub-pixel convolution) avoids this entirely. Instead of upsampling spatially and then convolving, it first convolves at the original low resolution to produce r² times more channels (where r is the upscale factor), and then rearranges those channels into spatial positions. For a 2x upscale, a 64-channel feature map becomes a 256-channel map via convolution, then PixelShuffle rearranges those 256 channels into a spatial grid that's 2x larger with 64 channels.
>
> I do this twice (2x → 2x = 4x total) instead of a single 4x jump because a single PixelShuffle(4) would require 64 × 16 = 1024 intermediate channels, which is memory-expensive. Two stages of PixelShuffle(2) need only 256 channels each — much more efficient."

### Common Mistakes
- Not knowing what checkerboard artifacts are.
- Not explaining why 2×2 is better than a single 4× jump.

### Follow-up Questions
1. "Can you show the math for how PixelShuffle rearranges channels to spatial positions?"
2. "What about bilinear upsampling followed by convolution? How does that compare?"
3. "Does PixelShuffle have any disadvantages?"

---

## Q5: Why did you keep the model at only 64 feature channels?

### Why the interviewer is asking this
They're testing your understanding of the efficiency vs performance tradeoff.

### Excellent Answer
> "It was a deliberate engineering decision driven by three constraints.
>
> First, deployment. I wanted the model to run inference in real-time on a CPU inside a Streamlit application. Doubling to 128 channels would quadruple the parameter count (since convolution parameters scale as C_in × C_out × K × K) and roughly 4x the inference time.
>
> Second, the dataset size. I only had 3,928 training patches. A 128-channel model with ~4M parameters would be far more prone to overfitting on this relatively small dataset. The 930K parameter model has a much better ratio of parameters to training samples.
>
> Third, diminishing returns. In super-resolution literature, going from 32 to 64 channels typically gives a significant PSNR boost (1-2 dB), but going from 64 to 128 usually yields only 0.1-0.3 dB improvement. The cost-benefit doesn't justify it for my use case."

### Common Mistakes
- Saying "I just picked 64 because tutorials use it."
- Not connecting the channel count to overfitting risk given the dataset size.

### Follow-up Questions
1. "What's the exact parameter count formula for your architecture?"
2. "If you had 100,000 training images, would you increase the channels?"
3. "Have you experimented with 32 channels? What happened?"

---

## Q6: Why is your model called Swin2SR in the code but it's actually a CNN?

### Why the interviewer is asking this
If they read your GitHub, they'll see this. It tests your honesty and engineering pragmatism.

### Excellent Answer
> "Great catch. The class is named `Swin2SR` for backwards compatibility, but the actual architecture is a Deep Residual Attention Network — a pure CNN. 
>
> During development, I initially planned to implement the actual Swin Transformer-based Swin2SR architecture. However, after analyzing the computational requirements, I realized that the original Swin2SR has around 12 million parameters and requires significantly more GPU memory and training time. Given my dataset size of only 3,928 patches and my compute budget on Kaggle's free T4 GPUs, a Transformer would have been severely overfitting.
>
> So I pivoted to a more efficient CNN architecture but kept the class name to avoid breaking the import chain in my Streamlit application and training scripts. In hindsight, I should have renamed it — it's a technical debt item I'd clean up in a refactor."

### Common Mistakes
- Trying to hide it or pretending it's actually a Transformer.
- Not acknowledging it as technical debt.

### Follow-up Questions
1. "What specific changes would you need to implement actual Swin2SR?"
2. "At what dataset size would a Transformer start outperforming your CNN?"
3. "What is the key difference between CNN attention (SE blocks) and Transformer self-attention?"
