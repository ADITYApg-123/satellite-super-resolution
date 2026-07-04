# 05 — Data Engineering

Interviewers love asking about data because it reveals whether you can handle messy, real-world inputs — not just clean Kaggle CSVs.

---

## Q1: Tell me about the dataset you used.

### Why the interviewer is asking this
Data is the foundation. If you don't deeply understand your data, nothing else matters.

### Excellent Answer
> "I used the WorldStrat dataset, which is specifically designed for satellite image super-resolution. It provides co-registered pairs of low-resolution Sentinel-2 imagery (10 meters per pixel) and high-resolution Airbus SPOT 6/7 imagery (1.5 meters per pixel).
>
> The dataset covers 3,928 geographic locations across the globe — not just wealthy Western cities, but also refugee settlements, deforested regions, industrial zones, and agricultural areas. This geographic diversity is critical because a model trained only on European cities would fail catastrophically on African farmland or Arctic tundra.
>
> The Sentinel-2 images come as 12-channel multispectral GeoTIFFs containing bands like Coastal Aerosol, Blue, Green, Red, Vegetation Red Edge, Near Infrared, and others. The SPOT images come as 12-bit pansharpened TIFFs. Each location has multiple temporal captures, meaning we get several images of the same place taken on different dates."

### Common Mistakes
- Saying "I used a satellite dataset" without naming it or knowing its properties.
- Not knowing that Sentinel-2 has 12 bands (this is a key technical detail).
- Not explaining why co-registration matters.

### Follow-up Questions
1. "What is co-registration and why is it critical for supervised SR?"
2. "How many spectral bands does Sentinel-2 have? Can you name them?"
3. "Why didn't you use all 12 bands? Wouldn't more information help?"
4. "What is the spatial resolution of each Sentinel-2 band?"

---

## Q2: Your data was 12-channel multispectral. How did you handle that?

### Why the interviewer is asking this
This is a practical data engineering question that tests real-world problem-solving.

### Excellent Answer
> "Standard image libraries like OpenCV and PIL are designed for RGB (3-channel) or RGBA (4-channel) images. When I first tried loading the Sentinel-2 GeoTIFFs with `cv2.imread()`, it crashed with a C++ decoder error: 'Unsupported number of channels: channels >= 1 && channels <= 4 where channels is 12.'
>
> I replaced OpenCV with `rasterio`, a geospatial raster I/O library built on GDAL. Rasterio can read arbitrary-channel TIFFs natively. After loading the 12-channel array, I extracted only bands [3, 2, 1] — which correspond to Band 4 (Red), Band 3 (Green), and Band 2 (Blue) in the Sentinel-2 specification — to construct a standard true-color RGB image.
>
> I chose to use only RGB because my target application (visual super-resolution for human viewing) requires visible-spectrum output. However, the architecture is designed so that changing the input from 3 to 12 channels requires modifying only a single line: `nn.Conv2d(3, 64, ...)` to `nn.Conv2d(12, 64, ...)`. If I wanted to leverage the infrared and vegetation bands for agricultural applications, the model could be retrained with minimal code changes."

### Common Mistakes
- Not knowing which bands correspond to RGB in Sentinel-2.
- Saying "I converted it to RGB" without explaining how.
- Not mentioning the extensibility to all 12 bands.

### Follow-up Questions
1. "Why bands [3, 2, 1] and not [0, 1, 2]?"
2. "Would using Near Infrared improve super-resolution quality?"
3. "What is GDAL and how does rasterio relate to it?"
4. "How did you normalize the pixel values? Were they 12-bit or 8-bit?"

---

## Q3: How did you split the data for training and validation?

### Why the interviewer is asking this
Data leakage is a common mistake. They want to know if your validation is rigorous.

### Excellent Answer
> "I used a 90/10 random split via PyTorch's `random_split()`, giving me approximately 3,535 training patches and 393 validation patches.
>
> The key consideration for satellite data is geographic leakage. If I extract multiple patches from the same geographic scene, and some end up in training while others end up in validation, the model could memorize location-specific features and appear to generalize when it's actually overfitting. WorldStrat mitigates this because each of the 3,928 entries represents a distinct geographic location — they're not overlapping crops from the same large image.
>
> I did not use a separate test set because the dataset was relatively small. In a production setting, I would reserve 10% for validation and 10% for a held-out test set that's only evaluated once at the very end."

### Common Mistakes
- Not mentioning geographic leakage as a concern.
- Using a test set and validation set interchangeably.

### Follow-up Questions
1. "What is data leakage? Can you give an example in satellite imagery?"
2. "Would k-fold cross-validation be appropriate here? Why or why not?"
3. "How would you create a geographically stratified split?"

---

## Q4: What data augmentations did you apply?

### Why the interviewer is asking this
Augmentation strategy reveals domain knowledge. Satellite images require different augmentations than face photos.

### Excellent Answer
> "I applied geometric augmentations: random rotations of 90°, 180°, and 270°, plus horizontal and vertical flips. These are ideal for satellite imagery because there's no 'correct' orientation — a building looks the same from any compass direction when viewed from directly above.
>
> I deliberately avoided several augmentations that are common in natural image processing. Color jittering is dangerous because in remote sensing, colors have physical meaning — the ratio of red to near-infrared reflectance indicates vegetation health. Randomly shifting colors would destroy this information. Random cropping at different scales is also risky because it would change the effective spatial resolution, confusing the model about what '10 meters per pixel' means.
>
> The geometric augmentations effectively multiply the dataset by 8x (4 rotations × 2 flip states), giving us approximately 31,000 effective training samples from our 3,928 base patches."

### Common Mistakes
- Applying augmentations that are valid for natural images but invalid for satellite data.
- Not knowing why color jittering is dangerous for multispectral data.

### Follow-up Questions
1. "Would you apply the same augmentation to both the LR and HR image in a pair?"
2. "What about elastic deformations or CutMix?"
3. "Could you use synthetic data generation instead of augmentation?"

---

## Q5: How did you handle the high-resolution SPOT imagery?

### Why the interviewer is asking this
The HR data has its own quirks. This tests thoroughness.

### Excellent Answer
> "The SPOT 6/7 imagery in WorldStrat is stored as 12-bit pansharpened TIFFs inside a nested directory structure: `hr_dataset/12bit/`. Each location folder contains four different files: panchromatic, pansharpened, RGB composite, and a metadata file.
>
> During initial setup on Kaggle, the DataLoader found zero matches between LR and HR images. I wrote a diagnostic script to recursively scan the file system and discovered the hidden nested structure. The fix was to explicitly map the DataLoader to target only the `_ps.tiff` (pansharpened) files, which are the 3-channel HR ground truth images.
>
> The pansharpened images are 12-bit, meaning pixel values range from 0 to 4095 instead of the standard 0-255. I normalized them to [0, 1] float range by dividing by 4095. This normalization must be consistent between LR and HR pairs — if you normalize one by 255 and the other by 4095, the model learns nonsensical mappings."

### Common Mistakes
- Not knowing what "pansharpened" means.
- Assuming all images use 8-bit depth.

### Follow-up Questions
1. "What does pansharpening mean in remote sensing?"
2. "What's the advantage of 12-bit over 8-bit for training?"
3. "How did you ensure the LR and HR images were properly aligned?"
