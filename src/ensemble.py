import numpy as np

def adaptive_ensemble_fusion(pred_a, pred_b, alpha=0.7):
    """
    Fuses the predictions from two different super-resolution models.
    For example:
    - pred_a: Swin2SR (Great at high-frequency details, but can hallucinate)
    - pred_b: Real-ESRGAN or Bicubic (More conservative, fewer details but safer)
    
    Alpha controls the blend. 0.7 means 70% Swin2SR, 30% conservative model.
    """
    return (alpha * pred_a) + ((1.0 - alpha) * pred_b)

def generate_gaussian_window(window_size):
    """
    Creates a 2D Hanning window (a bell curve in 2D space).
    When we stitch our 64x64 satellite patches back into a massive city-scale image,
    the hard edges of the squares normally create ugly "checkerboard" seam artifacts.
    
    By multiplying each patch by this Gaussian window, the center of the patch stays 
    bright (weight = 1.0) but the edges fade softly to 0.0. When overlapping patches 
    are added together, they blend seamlessly.
    """
    # Create a 1D Hanning window (bell curve)
    hanning_1d = np.hanning(window_size)
    
    # Take the outer product to make it 2D
    hanning_2d = np.outer(hanning_1d, hanning_1d)
    
    # Expand dims to match (H, W, C) for broadcasting
    return np.expand_dims(hanning_2d, axis=-1)

def accumulate_blended_tile(canvas, counts, patch, coords, gaussian_window):
    """
    Places an upscaled, Gaussian-weighted patch back onto the master canvas.
    This replaces the naive "reconstruct_from_windows" we wrote in Stage 2.
    """
    y, x = coords
    h, w, c = patch.shape
    
    # Multiply the patch by the soft-edge window
    weighted_patch = patch * gaussian_window
    
    # Add the weighted pixels to the master canvas
    canvas[y:y+h, x:x+w, :] += weighted_patch
    
    # Add the weights to the count canvas (so we can properly average the overlaps later)
    counts[y:y+h, x:x+w, :] += gaussian_window
    
    return canvas, counts
