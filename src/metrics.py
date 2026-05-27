import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def compute_psnr(sr_image, hr_image):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between Super-Resolved and Ground Truth.
    Higher is better (>30 dB is excellent for SR).
    Inputs should be numpy arrays in range [0, 1].
    """
    # Using data_range=1.0 because our normalization maps pixels from 0 to 1.0
    return compare_psnr(hr_image, sr_image, data_range=1.0)

def compute_ssim(sr_image, hr_image):
    """
    Computes Structural Similarity Index (SSIM).
    Measures human-perceived visual quality (luminance, contrast, structure).
    Higher is better (closer to 1.0).
    Inputs should be numpy arrays in range [0, 1] with shape (H, W, C).
    """
    # channel_axis=-1 specifies that color channels are the last dimension
    return compare_ssim(hr_image, sr_image, data_range=1.0, channel_axis=-1)

def compute_hallucination_score(sr_tensor, lr_tensor, scale_factor=8):
    """
    Our custom safety metric against generative hallucinations.
    1. Mathematically degrades the generated SR output back to LR dimensions.
    2. Computes the SSIM between this degraded image and the original LR input.
    3. Returns 1.0 - SSIM.
    
    A high score means the model invented features that don't exist in the original image.
    Lower is safer.
    """
    # 1. Downsample SR back to LR dimensions using bicubic interpolation
    # PyTorch interpolate expects (B, C, H, W)
    degraded_sr = F.interpolate(
        sr_tensor, 
        scale_factor=1.0/scale_factor, 
        mode='bicubic', 
        align_corners=False
    )
    
    # 2. Convert PyTorch tensors to NumPy arrays for scikit-image metrics
    # Shape becomes (H, W, C)
    degraded_numpy = degraded_sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    lr_numpy = lr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # 3. Calculate SSIM between the downsampled prediction and the real low-res input
    ssim_val = compare_ssim(lr_numpy, degraded_numpy, data_range=1.0, channel_axis=-1)
    
    # 4. Return the Hallucination Score (Inverted SSIM)
    return 1.0 - ssim_val
