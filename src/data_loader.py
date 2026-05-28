import os
import time
import ee
import numpy as np
import pandas as pd
import torch
import cv2
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def normalize_sentinel_radiometry(image_array):
    """
    Sentinel-2 sensors record light in 16-bit integers (0 to 10000+).
    Deep Learning models expect floats between 0.0 and 1.0.
    
    Instead of min-max scaling (which gets distorted by bright clouds),
    we clip at 3000 (standard agricultural/urban max reflectance) and divide by 3000.
    """
    # Clip extreme bright spots (like clouds or metal roofs) to 3000
    clipped = np.clip(image_array, 0, 3000)
    # Normalize to 0.0 - 1.0
    normalized = clipped / 3000.0
    return normalized.astype(np.float32)

def fetch_gee_image_with_retry(roi, start_date, end_date, max_cloud_cover=10, max_retries=5):
    """
    Fetches Sentinel-2 Harmonized surface reflectance data from Google Earth Engine.
    Implements Exponential Backoff to prevent API rate-limit bans from Google.
    """
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
                  .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    retries = 0
    while retries < max_retries:
        try:
            # Attempt to get the first (least cloudy) image
            image = collection.first()
            if image is None:
                raise ValueError("No images found in GEE for the given criteria.")
            
            # Select RGB bands (B4=Red, B3=Green, B2=Blue)
            rgb_image = image.select(['B4', 'B3', 'B2'])
            return rgb_image
            
        except Exception as e:
            retries += 1
            wait_time = (2 ** retries)  # Exponential backoff: 2s, 4s, 8s, 16s
            print(f"GEE API error: {e}. Retrying in {wait_time} seconds... ({retries}/{max_retries})")
            time.sleep(wait_time)
            
    raise Exception("Max retries exceeded. Could not fetch from Google Earth Engine.")

# ==========================================
# PHASE 3 UPGRADES: WorldStrat Kaggle Loader
# ==========================================

class WorldStratDataset(Dataset):
    """
    V2 PyTorch Dataset for WorldStrat Kaggle Dataset.
    Dynamically loads LR and HR image pairs from disk using the metadata.csv manifest.
    """
    def __init__(self, root_dir, csv_file='metadata.csv', max_cloud_cover=10.0, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        
        # Load the CSV manifest
        csv_path = os.path.join(root_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Operating in dummy mode for local testing.")
            self.metadata = pd.DataFrame()
            return
            
        self.metadata = pd.read_csv(csv_path)
        
        # Filter out cloudy images (assuming a 'cloud_cover' column exists)
        if 'cloud_cover' in self.metadata.columns:
            self.metadata = self.metadata[self.metadata['cloud_cover'] < max_cloud_cover]
            
        # Reset index after filtering
        self.metadata = self.metadata.reset_index(drop=True)
        print(f"Loaded WorldStrat Dataset: {len(self.metadata)} pristine image pairs found.")

    def __len__(self):
        if self.metadata.empty:
            return 8  # Return a few dummy batches for local testing
        return len(self.metadata)
        
    def _apply_augmentations(self, lr_tensor, hr_tensor):
        """
        Applies safe geometric augmentations. 
        Crucial: Exact same augmentations must be applied to BOTH the LR and HR image!
        """
        # Random Horizontal Flip
        if random.random() > 0.5:
            lr_tensor = TF.hflip(lr_tensor)
            hr_tensor = TF.hflip(hr_tensor)
            
        # Random Vertical Flip
        if random.random() > 0.5:
            lr_tensor = TF.vflip(lr_tensor)
            hr_tensor = TF.vflip(hr_tensor)
            
        # Random 90-degree Rotation
        if random.random() > 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            lr_tensor = TF.rotate(lr_tensor, angle)
            hr_tensor = TF.rotate(hr_tensor, angle)
            
        return lr_tensor, hr_tensor

    def __getitem__(self, idx):
        if self.metadata.empty:
            # Dummy mode for local IDE testing (prevents crashes before Kaggle deployment)
            # Returns a 3x64x64 LR and 3x256x256 HR (4x upscale)
            return torch.rand(3, 64, 64), torch.rand(3, 256, 256)
            
        row = self.metadata.iloc[idx]
        
        # Auto-detect column names dynamically to prevent KeyErrors
        if not hasattr(self, 'lr_col'):
            cols = [c.lower() for c in row.keys()]
            # Find the column that contains 'lr' or 'low'
            self.lr_col = next((c for c in row.keys() if 'lr' in c.lower() or 'low' in c.lower()), row.keys()[0])
            self.hr_col = next((c for c in row.keys() if 'hr' in c.lower() or 'high' in c.lower()), row.keys()[1])
            print(f"Auto-detected columns: LR={self.lr_col}, HR={self.hr_col}")
            
        lr_full_path = os.path.join(self.root_dir, row[self.lr_col])
        hr_full_path = os.path.join(self.root_dir, row[self.hr_col])
        
        # Load images (cv2.IMREAD_UNCHANGED reads 16-bit TIFFs correctly)
        lr_img = cv2.imread(lr_full_path, cv2.IMREAD_UNCHANGED)
        hr_img = cv2.imread(hr_full_path, cv2.IMREAD_UNCHANGED)
        
        # Convert BGR to RGB (OpenCV default is BGR)
        if len(lr_img.shape) == 3 and lr_img.shape[2] == 3:
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            
        # Normalize (assuming 16-bit Sentinel-2 data)
        lr_norm = normalize_sentinel_radiometry(lr_img)
        hr_norm = normalize_sentinel_radiometry(hr_img)
        
        # Convert to PyTorch tensors: (H, W, C) -> (C, H, W)
        lr_tensor = torch.from_numpy(lr_norm).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_norm).permute(2, 0, 1).float()
        
        # Apply data augmentations if in training mode
        if self.is_train:
            lr_tensor, hr_tensor = self._apply_augmentations(lr_tensor, hr_tensor)
            
        return lr_tensor, hr_tensor
