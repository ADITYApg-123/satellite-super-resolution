# 01 — Project Overview

These are the questions every single interviewer will start with. Nail these and you set the tone for the entire interview.

---

## Q1: Tell me about your project.

### Why the interviewer is asking this
They want to see if you can explain a complex ML system clearly and concisely. This is a communication test disguised as a technical question. If you ramble, they assume you don't truly understand what you built.

### Excellent Answer
> "GeoSafe SR is a deep learning system I built for satellite image super-resolution. The core problem is that freely available satellite imagery from Sentinel-2 has a resolution of only 10 meters per pixel, which is too blurry for practical applications like urban planning or disaster response. Commercial high-resolution satellite data costs thousands of dollars per image.
>
> I engineered a lightweight 930K-parameter Convolutional Neural Network that takes a blurry 10m/pixel Sentinel-2 image and produces a sharp 1.5m/pixel output — a 4x enhancement. The model was trained on 3,928 paired image patches from the WorldStrat dataset and achieved a peak PSNR of 30.14 dB.
>
> The unique engineering challenge was preventing the model from hallucinating fake structures. Standard super-resolution models using GANs will invent fake buildings or roads to make the image look sharper. I designed a custom composite loss function called GeoSafeLoss that mathematically constrains the GAN to only sharpen details that actually exist in the original data."

### Common Mistakes
- Jumping straight into architecture details without explaining the problem first.
- Saying "I used PyTorch and CNNs" without explaining WHY.
- Forgetting to mention the hallucination problem (this is what makes your project unique).

### Follow-up Questions
1. "What do you mean by hallucination in this context?"
2. "Why not just buy high-resolution satellite imagery?"
3. "How do you know the model isn't hallucinating?"
4. "What's PSNR and why did you choose it as your primary metric?"
5. "Who would actually use this in production?"

### Interviewer's Evaluation
**Strong signal**: Candidate structures the answer as Problem → Solution → Unique Challenge → Result. Mentions specific numbers.
**Weak signal**: Candidate says "I built a CNN to upscale images" with no context on why it matters.

---

## Q2: What problem does this project solve?

### Why the interviewer is asking this
They want to know if you understand the real-world context, not just the code. Engineers who only understand code but not the domain are less valuable.

### Excellent Answer
> "There's a massive gap in satellite imagery accessibility. Government agencies like ISRO or NASA provide free satellite data through missions like Sentinel-2, but this data is at 10 meters per pixel — you can see a football field, but you can't see individual buildings or roads clearly. High-resolution imagery at 1-2 meters per pixel exists from commercial providers like Maxar or Airbus, but it costs hundreds to thousands of dollars per scene.
>
> My project bridges this gap using AI. It takes the free, low-resolution data and computationally enhances it to near-commercial quality. This is valuable for disaster response teams who need immediate high-res imagery after an earthquake, urban planners tracking illegal construction, or environmental researchers monitoring deforestation — all without the massive cost of commercial satellite subscriptions."

### Common Mistakes
- Saying "super-resolution makes images bigger" (it's not about size, it's about recovering detail).
- Not mentioning the cost angle (this is what makes it practically relevant).

### Follow-up Questions
1. "How does this compare to just using Google Earth?"
2. "Can this replace commercial satellite imagery entirely?"
3. "What are the limitations of AI-generated super-resolution vs real high-res capture?"

---

## Q3: Why did you choose this problem specifically?

### Why the interviewer is asking this
They're evaluating your intellectual curiosity and self-motivation. Did you just copy a tutorial, or did you identify a real gap?

### Excellent Answer
> "Two reasons. First, I noticed that most super-resolution research focuses on natural photographs — faces, animals, landscapes — where hallucination is acceptable or even desirable. But in satellite imagery, hallucination is catastrophic. If the AI invents a road that doesn't exist, and a disaster response team uses that image for navigation, people could die. I found that fascinating as an engineering constraint.
>
> Second, I wanted a project that forced me to go beyond standard PyTorch tutorials. This project required me to learn geospatial data formats (12-channel multispectral TIFFs), custom loss function design, GAN training dynamics, and cloud GPU orchestration on Kaggle. It pushed me into territory that most junior engineers never touch."

### Common Mistakes
- Saying "I thought it would look good on my resume" (even if true, never say this).
- Not explaining what makes satellite SR different from regular SR.

### Follow-up Questions
1. "What's the difference between satellite SR and regular image SR?"
2. "Have you read any papers on this topic? Which ones influenced your design?"
3. "If hallucination is so dangerous, how do you quantify it?"

---

## Q4: What makes your project unique compared to existing solutions?

### Why the interviewer is asking this
They want to see if you've done competitive analysis. Can you position your work relative to the field?

### Excellent Answer
> "Three things differentiate it. First, the GeoSafeLoss function. Most satellite SR papers use standard L1 or MSE loss and get blurry results, or they use a GAN and get hallucinations. I engineered a 4-component loss function that gives you the sharpness of a GAN while mathematically constraining it to geographic reality using Sobel gradient penalties.
>
> Second, the efficiency. Models like Real-ESRGAN have 16 million parameters. Mine achieves comparable visual quality at 930K parameters — roughly 17x lighter. This matters for deployment on edge devices or in bandwidth-constrained disaster zones.
>
> Third, the end-to-end pipeline. This isn't just a model — it's a complete system. I built the geospatial data ingestion pipeline using rasterio, the training infrastructure with checkpoint resumption for cloud GPU timeouts, and a Streamlit application for real-time interactive inference."

### Common Mistakes
- Not knowing what Real-ESRGAN or EDSR are (you should know the baselines in your field).
- Claiming your model is "better than everything" (be honest about limitations).

### Follow-up Questions
1. "You said 17x lighter — does it perform 17x worse?"
2. "Have you benchmarked against Real-ESRGAN on the same dataset?"
3. "What would you need to change to deploy this on a drone or edge device?"
