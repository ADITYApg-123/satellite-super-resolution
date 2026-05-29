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
    WorldStrat Kaggle Dataset Loader.
    
    Dataset structure (verified from Kaggle):
      hr_dataset/12bit/{loc_id}/{loc_id}_ps.tiff   (pansharpened HR, ~1.5m/px)
      lr_dataset/{loc_id}/L2A/{loc_id}-N-L2A_data.tiff  (Sentinel-2 LR, ~10m/px)
    """
    def __init__(self, root_dir, csv_file='metadata.csv', is_train=True, patch_size=128):
        self.root_dir = root_dir
        self.is_train = is_train
        self.patch_size = patch_size
        self.hr_dir = os.path.join(root_dir, 'hr_dataset', '12bit')
        self.lr_dir = os.path.join(root_dir, 'lr_dataset')
        self.pairs = []
        
        # Scan hr_dataset/12bit/ for all location folders
        print(f"Scanning HR folder: {self.hr_dir}")
        if not os.path.exists(self.hr_dir):
            raise FileNotFoundError(f"HR folder not found: {self.hr_dir}")
        
        hr_locations = sorted(os.listdir(self.hr_dir))
        print(f"Found {len(hr_locations)} HR locations. Matching with LR...")
        
        for loc_id in hr_locations:
            # HR: use the pansharpened TIFF (_ps.tiff)
            hr_path = os.path.join(self.hr_dir, loc_id, f"{loc_id}_ps.tiff")
            
            # LR: use the first L2A_data revisit
            lr_l2a_dir = os.path.join(self.lr_dir, loc_id, 'L2A')
            lr_path = None
            
            if os.path.exists(lr_l2a_dir):
                for fname in sorted(os.listdir(lr_l2a_dir)):
                    if 'L2A_data' in fname and fname.endswith(('.tif', '.tiff')):
                        lr_path = os.path.join(lr_l2a_dir, fname)
                        break
            
            if os.path.exists(hr_path) and lr_path:
                self.pairs.append((lr_path, hr_path))
                
        if len(self.pairs) == 0:
            raise ValueError(f"No matching LR/HR pairs found!")
            
        print(f"✅ Successfully matched {len(self.pairs)} LR-HR image pairs!")

    def __len__(self):
        return len(self.pairs)
        
    def _apply_augmentations(self, lr_tensor, hr_tensor):
        if random.random() > 0.5:
            lr_tensor = TF.hflip(lr_tensor)
            hr_tensor = TF.hflip(hr_tensor)
        if random.random() > 0.5:
            lr_tensor = TF.vflip(lr_tensor)
            hr_tensor = TF.vflip(hr_tensor)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lr_tensor = TF.rotate(lr_tensor, angle)
            hr_tensor = TF.rotate(hr_tensor, angle)
        return lr_tensor, hr_tensor

import rasterio

    def _load_and_prepare(self, path, target_channels=3):
        """Load an image using rasterio (supports >4 channels)."""
        try:
            # rasterio reads as (C, H, W)
            with rasterio.open(path) as src:
                img = src.read()
            # Convert to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        except Exception:
            return None
        
        # Handle channel count
        if len(img.shape) == 2:
            img = np.stack([img] * target_channels, axis=-1)
        elif img.shape[2] > target_channels:
            # Sentinel-2 multi-spectral has 12 or 13 bands. 
            # True color RGB is Band 4 (Red), Band 3 (Green), Band 2 (Blue), which are indices 3, 2, 1.
            if img.shape[2] >= 12:
                img = img[:, :, [3, 2, 1]]
            else:
                img = img[:, :, :target_channels]
        elif img.shape[2] < target_channels:
            pad = np.zeros((*img.shape[:2], target_channels - img.shape[2]), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=-1)
        
        # Rasterio natively reads RGB (not BGR like OpenCV), so no cvtColor needed!
        return normalize_sentinel_radiometry(img)

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        
        hr_img = self._load_and_prepare(hr_path)
        lr_img = self._load_and_prepare(lr_path)
        
        # Skip broken images
        if hr_img is None or lr_img is None:
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))
        
        # Crop HR to a square patch
        h, w = hr_img.shape[:2]
        ps = min(self.patch_size, h, w)
        if self.is_train:
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
        else:
            top, left = (h - ps) // 2, (w - ps) // 2
        hr_patch = hr_img[top:top+ps, left:left+ps]
        
        # Create matching LR patch by downscaling HR by 4x
        lr_size = ps // 4
        lr_patch = cv2.resize(hr_patch, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        
        # To tensors: (H,W,C) -> (C,H,W)
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float()
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float()
        
        if self.is_train:
            lr_tensor, hr_tensor = self._apply_augmentations(lr_tensor, hr_tensor)
            
        return lr_tensor, hr_tensor

