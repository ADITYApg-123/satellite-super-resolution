import numpy as np

def extract_strided_windows(image_array, window_size=64, stride=32):
    """
    Slices a massive satellite image into small manageable chunks (windows) to prevent
    GPU Out-Of-Memory (OOM) errors during inference and training.
    
    Args:
        image_array: A numpy array of shape (H, W, C)
        window_size: The width/height of the extracted patch (e.g., 64 for Sentinel-2)
        stride: The step size between windows. If stride < window_size, patches overlap.
                Overlap is critical for preventing "seam" artifacts when reconstructing.
                
    Returns:
        List of (patch_array, coordinates) where coordinates are (y, x).
    """
    h, w, c = image_array.shape
    patches = []
    
    # Iterate over the image with the specified stride
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            # Extract the window
            patch = image_array[y:y+window_size, x:x+window_size, :]
            patches.append((patch, (y, x)))
            
    return patches

def reconstruct_from_windows(patches_with_coords, original_h, original_w, window_size, scale_factor=8):
    """
    Rebuilds the massive High-Resolution image from the upscaled HR patches.
    (This is a stub function. In Stage 5, we will upgrade this to use Gaussian
    Alpha Blending to seamlessly merge the overlapping edges).
    """
    # Calculate the new HR dimensions
    hr_h = original_h * scale_factor
    hr_w = original_w * scale_factor
    hr_window = window_size * scale_factor
    
    # Create an empty canvas and a count canvas (for averaging overlapping pixels)
    canvas = np.zeros((hr_h, hr_w, 3), dtype=np.float32)
    counts = np.zeros((hr_h, hr_w, 3), dtype=np.float32)
    
    for patch, (y, x) in patches_with_coords:
        # Scale the coordinates up to HR space
        hr_y = y * scale_factor
        hr_x = x * scale_factor
        
        # Add the patch to the canvas and increment the count
        canvas[hr_y:hr_y+hr_window, hr_x:hr_x+hr_window, :] += patch
        counts[hr_y:hr_y+hr_window, hr_x:hr_x+hr_window, :] += 1.0
        
    # Prevent division by zero
    counts[counts == 0] = 1.0
    
    # Average overlapping areas
    final_image = canvas / counts
    return final_image
