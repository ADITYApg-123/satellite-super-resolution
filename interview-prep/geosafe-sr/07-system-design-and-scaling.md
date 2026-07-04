# 07 — System Design and Scaling

These questions test whether you can think beyond a single-machine prototype. Even if your project is small, interviewers want to see systems thinking.

---

## Q1: How would you scale this system to process 10,000 images per hour?

### Why the interviewer is asking this
Classic system design question. They want to see if you can think about production architecture.

### Excellent Answer
> "The current Streamlit app processes images one at a time on a single CPU. To scale to 10,000 images per hour, I would redesign the system as follows:
>
> **Inference Server**: Replace Streamlit with a FastAPI backend that exposes a `/predict` REST endpoint. The model would be loaded once at startup and kept in GPU memory. Each request sends an image, gets back the super-resolved result.
>
> **Batching**: Instead of processing one image per request, I would implement a request queue that accumulates images and batches them together. GPU utilization is much higher when processing a batch of 8 or 16 images simultaneously versus one at a time.
>
> **Horizontal Scaling**: Deploy multiple inference containers behind a load balancer. Each container has its own GPU and model copy. An NVIDIA Triton Inference Server or TorchServe would handle model serving, batching, and GPU scheduling automatically.
>
> **Async Processing**: For large jobs, I would use a task queue (like Celery with Redis) where the client submits an image and receives a job ID. A worker picks up the job, processes it, and stores the result in object storage (S3). The client polls for completion.
>
> **Model Optimization**: Before deploying, I would convert the model to TorchScript or ONNX for faster inference, and potentially apply INT8 quantization to reduce the model size from 3.5MB to ~1MB while maintaining acceptable quality."

### Common Mistakes
- Only mentioning "use more GPUs" without discussing batching, queuing, or model optimization.
- Not mentioning the difference between synchronous and asynchronous processing.

### Follow-up Questions
1. "How would you handle a spike of 100,000 requests in 1 minute?"
2. "What's the cost of running this on AWS with GPU instances?"
3. "How would you implement A/B testing between two model versions?"

---

## Q2: How would you deploy this model for edge/mobile inference?

### Why the interviewer is asking this
Edge deployment is increasingly important, especially for satellite applications in disaster zones with no internet.

### Excellent Answer
> "My model is already lightweight at 930K parameters, which makes it a strong candidate for edge deployment. Here's my approach:
>
> **Step 1 — Model Optimization**: Convert the PyTorch model to ONNX format, then use ONNX Runtime for optimized inference. For mobile specifically, I would use TensorFlow Lite or PyTorch Mobile.
>
> **Step 2 — Quantization**: Apply post-training INT8 quantization. This converts the 32-bit floating-point weights to 8-bit integers, reducing model size by 4x and inference time by 2-3x on CPUs. Since my model is relatively simple (no batch normalization, no complex attention), quantization should have minimal accuracy loss.
>
> **Step 3 — Hardware-Specific Optimization**: For NVIDIA Jetson devices (common in field deployments), I would use TensorRT to optimize the computational graph. For mobile phones, CoreML (iOS) or NNAPI (Android) can leverage the device's neural processing unit.
>
> **Key Concern**: The upsampling layers (PixelShuffle) can be memory-intensive because they temporarily create 256-channel feature maps. For severely memory-constrained devices, I might need to reduce this by processing the image in overlapping tiles and stitching the results."

### Common Mistakes
- Saying "just use a smaller model" without discussing optimization techniques.
- Not mentioning the memory implications of PixelShuffle on edge devices.

### Follow-up Questions
1. "What is the minimum hardware needed to run your model?"
2. "How much accuracy do you lose with INT8 quantization?"
3. "Could you run this on a Raspberry Pi?"

---

## Q3: If ISRO asked you to process all of India's Sentinel-2 data daily, how would you architect it?

### Why the interviewer is asking this
This is a scale-up design question specific to your domain. It tests both ML and infrastructure knowledge.

### Excellent Answer
> "India spans roughly 3.3 million km². Sentinel-2 captures a 290km-wide swath with a 5-day revisit period, so approximately 1/5th of India is captured daily. At 10m/pixel, each daily capture is enormous — potentially terabytes of raw data.
>
> **Data Ingestion**: Set up an automated pipeline using Google Earth Engine or the Copernicus Open Access Hub API to pull new Sentinel-2 tiles as they become available. Store raw tiles in cloud object storage (S3 or GCS).
>
> **Preprocessing Pipeline**: Use Apache Beam or AWS Step Functions to orchestrate the preprocessing: cloud masking (removing cloudy tiles), atmospheric correction, tiling into 128x128 patches, and normalization. This runs on CPU-only auto-scaling worker pools.
>
> **Inference Pipeline**: Deploy the model on a fleet of GPU instances (e.g., AWS g4dn.xlarge with T4 GPUs) managed by Kubernetes. Use a job queue to distribute patches across workers. With batched inference, each GPU could process ~1000 patches per minute.
>
> **Post-processing**: Stitch the super-resolved patches back into full tiles, applying overlap-and-blend at seam boundaries to prevent visible tile edges. Store results in a GeoTIFF pyramid for multi-resolution access.
>
> **Monitoring**: Track inference latency, PSNR quality scores, GPU utilization, and queue depth. Set up alerts for quality degradation (which might indicate sensor calibration changes or seasonal distribution shifts)."

### Common Mistakes
- Not considering cloud masking (many satellite images are partially cloudy).
- Forgetting the stitching problem (tile boundary artifacts).

### Follow-up Questions
1. "How would you handle cloudy images?"
2. "What happens when Sentinel-2 changes its calibration? Would your model break?"
3. "How would you version your model in production?"

---

## Q4: What would you change about the architecture if you had unlimited compute?

### Why the interviewer is asking this
This tests your knowledge of the broader super-resolution landscape and your ability to think beyond your current constraints.

### Excellent Answer
> "Three major changes.
>
> First, I would replace the CNN backbone with a Vision Transformer like SwinIR or HAT (Hybrid Attention Transformer). Transformers can capture long-range dependencies that CNNs miss — for example, the relationship between a river on one side of the image and the vegetation pattern on the other side. This would require 10-50x more parameters but would likely push PSNR above 32 dB.
>
> Second, I would implement multi-temporal input. Instead of feeding a single Sentinel-2 image, I would feed 3-5 images of the same location taken on different dates. A temporal fusion module would learn to combine information across timestamps — using clear pixels from one date to fill in cloudy or blurry regions from another. This is one of the most promising directions in satellite SR research.
>
> Third, I would add a diffusion-based refinement stage. After the main model produces a 4x upscaled image, a lightweight diffusion model would iteratively denoise and sharpen the output. Recent papers like StableSR show that diffusion models excel at adding realistic high-frequency texture detail."

### Common Mistakes
- Just saying "I would make it bigger" without naming specific architectures.
- Not mentioning multi-temporal fusion (this shows domain expertise).

### Follow-up Questions
1. "What is the key difference between CNN attention and Transformer self-attention?"
2. "How would you handle temporal misalignment between multi-date images?"
3. "What is the computational complexity of self-attention vs convolution?"
