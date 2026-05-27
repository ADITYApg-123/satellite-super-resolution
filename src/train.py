import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_swin2sr import initialize_swin2sr_model
from metrics import compute_psnr, compute_ssim

def train_swin2sr(dataset, epochs=10, batch_size=4, learning_rate=2e-4, device='cuda'):
    """
    The core training engine for Fine-Tuning our Swin2SR architecture.
    Designed to be run on Kaggle (T4/P100 GPUs) using a pre-curated dataset.
    """
    print(f"Initializing Training Pipeline on {device}...")
    
    # 1. Initialize Model & move to GPU
    model = initialize_swin2sr_model()
    model = model.to(device)
    model.train() # Set to training mode
    
    # 2. Setup Data Loader
    # In Kaggle, 'dataset' will be populated with tens of thousands of WorldStrat image pairs
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 3. Define Optimizer and Loss
    # AdamW is industry standard for Transformers (handles weight decay better than Adam)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # L1 Loss (Mean Absolute Error) is preferred over MSE for Super-Resolution
    # because MSE penalizes large errors heavily, leading to blurry, overly-smooth images.
    # L1 preserves sharp, high-frequency edges.
    criterion = nn.L1Loss()
    
    # 4. Initialize Mixed Precision Scaler
    # This is critical for Kaggle GPUs. It calculates math in 16-bit float where possible,
    # cutting VRAM usage in half and speeding up training by 30-40%.
    scaler = torch.cuda.amp.GradScaler()
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Wrap dataloader in tqdm for a nice progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(pbar):
            # Move batches to GPU
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast():
                # Get the Super-Resolved prediction
                sr_prediction = model(lr_batch)
                
                # Calculate L1 error between our prediction and the real High-Res image
                loss = criterion(sr_prediction, hr_batch)
                
            # Backward pass (calculate gradients using the scaler)
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            # Transformers are prone to exploding gradients during early epochs. 
            # This clips massive gradients to a safe threshold (e.g., 1.0)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"L1 Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        
        # Save a checkpoint every epoch
        torch.save(model.state_dict(), f"swin2sr_epoch_{epoch+1}.pth")
        print(f"Saved Checkpoint: swin2sr_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    print("Train engine ready. Import this script in your Kaggle notebook to begin training.")
