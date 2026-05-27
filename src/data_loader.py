import os
import time
import ee
import numpy as np
import torch
from torch.utils.data import Dataset

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

class WorldStratDataset(Dataset):
    """
    PyTorch Dataset class for loading satellite patches.
    In a full training loop, this will iteratively load our 64x64 chunks.
    """
    def __init__(self, patch_list):
        self.patches = patch_list
        
    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        # Retrieve the numpy array
        patch_array, coords = self.patches[idx]
        
        # Normalize the 16-bit radiometry to 0.0-1.0 floats
        normalized_patch = normalize_sentinel_radiometry(patch_array)
        
        # PyTorch expects (Channels, Height, Width), but images are usually (H, W, C)
        tensor_patch = torch.from_numpy(normalized_patch).permute(2, 0, 1)
        return tensor_patch, coords
