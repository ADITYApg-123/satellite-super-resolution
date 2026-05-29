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
        self.hr_dir = os.path.join(root_dir, 'hr_dataset')
        self.lr_dir = os.path.join(root_dir, 'lr_dataset')
        
        csv_path = os.path.join(root_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find CSV at {csv_path}")
            
        self.metadata = pd.read_csv(csv_path)
        self.pairs = []
        
        # The first column 'Unnamed: 0' contains the Location ID (e.g. 'Landcover-777684')
        id_col = self.metadata.columns[0]
        
        print(f"Scanning dataset folders in {root_dir}...")
        for loc_id in self.metadata[id_col]:
            loc_id = str(loc_id)
            hr_path = None
            lr_path = None
            
            # 1. Find the High-Res Image
            hr_loc_dir = os.path.join(self.hr_dir, loc_id)
            if os.path.exists(hr_loc_dir):
                for f in os.listdir(hr_loc_dir):
                    if f.endswith('.tif') or f.endswith('.tiff'):
                        hr_path = os.path.join(hr_loc_dir, f)
                        break
                        
            # 2. Find the Low-Res Image (Sentinel-2 L2A data)
            lr_l2a_dir = os.path.join(self.lr_dir, loc_id, 'L2A')
            if os.path.exists(lr_l2a_dir):
                for f in os.listdir(lr_l2a_dir):
                    if 'L2A_data.tif' in f or 'L2A_data.tiff' in f:
                        lr_path = os.path.join(lr_l2a_dir, f)
                        break
            
            if hr_path and lr_path:
                self.pairs.append((lr_path, hr_path))
                
        if len(self.pairs) == 0:
            raise ValueError(f"CRITICAL ERROR: Could not find any matching HR and LR images in {root_dir}")
            
        print(f"Successfully matched {len(self.pairs)} image pairs!")

    def __len__(self):
        return len(self.pairs)
        
    def _apply_augmentations(self, lr_tensor, hr_tensor):
        """
        Applies safe geometric augmentations. 
        Crucial: Exact same augmentations must be applied to BOTH the LR and HR image!
        """
        if random.random() > 0.5:
            lr_tensor = TF.hflip(lr_tensor)
            hr_tensor = TF.hflip(hr_tensor)
            
        if random.random() > 0.5:
            lr_tensor = TF.vflip(lr_tensor)
            hr_tensor = TF.vflip(hr_tensor)
            
        if random.random() > 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            lr_tensor = TF.rotate(lr_tensor, angle)
            hr_tensor = TF.rotate(hr_tensor, angle)
            
        return lr_tensor, hr_tensor

    def __getitem__(self, idx):
        lr_full_path, hr_full_path = self.pairs[idx]
        
        # Load images (cv2.IMREAD_UNCHANGED reads 16-bit TIFFs correctly)
        lr_img = cv2.imread(lr_full_path, cv2.IMREAD_UNCHANGED)
        hr_img = cv2.imread(hr_full_path, cv2.IMREAD_UNCHANGED)
        
        if lr_img is None:
            raise ValueError(f"Failed to load LR image: {lr_full_path}")
        if hr_img is None:
            raise ValueError(f"Failed to load HR image: {hr_full_path}")
        
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
