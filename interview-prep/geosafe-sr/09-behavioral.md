# 09 — Behavioral Questions

These questions assess your character, self-awareness, and growth mindset. They appear in EVERY interview round. Don't wing them.

---

## Q1: Why did you build this project?

### Why the interviewer is asking this
They want to gauge intrinsic motivation. "For my resume" is the worst possible answer.

### Excellent Answer
> "I was frustrated by a gap I noticed in the super-resolution research community. Most SR work focuses on natural photographs — faces, landscapes, cityscapes — where the model is encouraged to hallucinate plausible textures because the goal is visual beauty. But in satellite imagery, hallucination isn't just wrong, it's dangerous. If the AI invents a road that doesn't exist and an emergency response team uses that image for routing, the consequences could be severe.
>
> I wanted to prove that you could build a system that gets the visual sharpness of a GAN while maintaining the geometric trustworthiness needed for real-world geographic applications. It was also a personal challenge — this project forced me to learn geospatial data formats, custom loss function design, GAN training dynamics, and cloud GPU management all at once."

### Common Mistakes
- Saying "I wanted to learn PyTorch" (too generic).
- Not connecting the project to a real-world impact.

---

## Q2: What's the biggest mistake you made during development?

### Why the interviewer is asking this
They want honesty and self-awareness. The best engineers openly discuss their failures.

### Excellent Answer
> "My biggest mistake was treating satellite imagery like regular photographs. I spent the first week of development loading images with OpenCV and testing with small PNG samples that happened to be 3-channel. Everything worked perfectly locally.
>
> When I pushed to Kaggle and hit the actual 12-channel Sentinel-2 GeoTIFFs, everything exploded. OpenCV's C++ backend crashed, the pixel values were in a completely different range (12-bit vs 8-bit), and the file structure was nothing like what I expected.
>
> I lost two full days debugging issues that I could have avoided entirely if I had started by loading and inspecting the real data format first. This taught me a critical lesson: in ML engineering, always start with the data. Write a simple script that loads one sample, prints its shape, dtype, value range, and visualizes it before writing a single line of model code."

### Common Mistakes
- Choosing a trivial mistake (like a typo).
- Not explaining what you learned from it.

---

## Q3: What's the one thing you're most proud of?

### Why the interviewer is asking this
They want to see what you value as an engineer. Your answer reveals your priorities.

### Excellent Answer
> "The GeoSafeLoss function. Specifically, the moment I realized that the Gradient Profile Loss with Sobel filters was the key to suppressing hallucination.
>
> I had spent days trying different weight combinations for L1 and GAN loss. The model either produced blurry images (L1 too strong) or hallucinated structures (GAN too strong). It felt like an impossible tradeoff. Then I added the Sobel-based edge loss, and suddenly the model could produce sharp images without inventing fake edges. The GAN could improve texture quality within the boundaries set by the edge constraint.
>
> I'm proud of it because it wasn't copied from a tutorial or a paper. I reasoned about the problem — 'What specific property distinguishes a hallucinated building from a real one? The edges.' — and then designed a loss component to exploit that distinction. It was genuine engineering, not code copying."

### Common Mistakes
- Being proud of something trivial (like "getting it to run").
- Not showing the thought process behind the achievement.

---

## Q4: What would you improve if you had two more months?

### Why the interviewer is asking this
They want to see if you can think ahead and identify the project's weaknesses honestly.

### Excellent Answer
> "Three things, in priority order.
>
> First, multi-temporal input. Right now the model sees a single snapshot. If that image is partially cloudy or has atmospheric haze, the model has to guess what's underneath. If I fed it 3-5 images of the same location from different dates, it could use clear pixels from one date to fill in degraded regions from another. This is one of the most promising directions in satellite SR research.
>
> Second, I would run proper ablation studies. I haven't systematically measured what happens when I remove each loss component individually. I believe the Gradient Profile Loss is critical, but I can't prove it with data yet. A proper ablation table would significantly strengthen both the project and my interview answers.
>
> Third, I would deploy it to Hugging Face Spaces with the model checkpoint hosted on Hugging Face Hub. Right now, recruiters have to clone the repo and run `pip install` to see the demo. A live link would dramatically improve the portfolio impact."

### Common Mistakes
- Saying "nothing, it's perfect" (arrogant and unrealistic).
- Listing improvements that show you don't understand the current limitations.

---

## Q5: What did this project teach you?

### Why the interviewer is asking this
This is the closing question. Your answer should leave a lasting impression.

### Excellent Answer
> "Three fundamental lessons.
>
> First, that loss function design is more impactful than architecture design. I could have spent weeks implementing the Swin Transformer, but instead I invested that time into engineering GeoSafeLoss. The result was a 930K parameter model that achieves competitive quality against models 17x its size. The loss function taught the model what to care about; the architecture just provided the capacity to learn it.
>
> Second, that data engineering is 60% of ML engineering. The most challenging part of this project wasn't the neural network — it was parsing 12-channel multispectral TIFFs, handling 12-bit pixel depths, navigating hidden directory structures, and ensuring co-registration between LR and HR pairs. In university, we focus on model architecture. In the real world, the model is the easy part.
>
> Third, that proper MLOps saves days of rework. Implementing a comprehensive checkpoint-resume system with full optimizer state persistence, adding `.detach()` for GAN memory management, and using mixed-precision training — these 'boring' engineering practices are what allowed me to actually finish the project instead of endlessly fighting GPU crashes and corrupted training runs."

### Common Mistakes
- Giving a generic answer like "I learned PyTorch."
- Not tying the lessons back to professional engineering principles.

---

## Q6: How did you handle this project being solo? Didn't you need a team?

### Why the interviewer is asking this
They want to know if you can work independently and manage scope.

### Excellent Answer
> "Being solo forced extreme discipline in scope management. Every feature I wanted to add had to pass a filter: 'Does this make the core pipeline better, or is it a distraction?'
>
> For example, I initially wanted to implement the full Swin2SR Transformer architecture. But I calculated that training it on my dataset size (3,928 patches) would likely cause severe overfitting, and debugging a Transformer on Kaggle's 12-hour GPU limits would be extremely painful. So I pivoted to a simpler CNN architecture that I could train, debug, and iterate on quickly.
>
> I also documented every decision and bug in a PROJECT_JOURNEY.md file — a running engineering log. This served as my 'team memory' since I couldn't rely on a colleague to remember why we made certain decisions three weeks ago.
>
> If I had a team, I would have split the work: one person on data engineering and augmentation, one on model architecture and training, and one on the Streamlit deployment and evaluation pipeline."

### Common Mistakes
- Saying "I did everything myself" without showing structured decision-making.
- Not acknowledging the limitations of working solo.
