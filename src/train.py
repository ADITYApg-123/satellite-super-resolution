import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from model_swin2sr import Swin2SR
from loss_functions import UNetDiscriminator, GeoSafeLoss
from metrics import compute_psnr, compute_ssim

def save_checkpoint(epoch, generator, discriminator, opt_g, opt_d, scheduler_g, scheduler_d, best_psnr, filename):
    """Saves the complete training state so we can resume if Kaggle kills the notebook."""
    checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': opt_g.state_dict(),
        'optimizer_D': opt_d.state_dict(),
        'scheduler_G': scheduler_g.state_dict(),
        'scheduler_D': scheduler_d.state_dict(),
        'best_psnr': best_psnr
    }
    torch.save(checkpoint, filename)
    print(f"Saved Checkpoint: {filename}")

def load_checkpoint(resume_path, generator, discriminator, opt_g, opt_d, scheduler_g, scheduler_d):
    """Loads the complete training state."""
    print(f"Resuming training from {resume_path}...")
    checkpoint = torch.load(resume_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    opt_g.load_state_dict(checkpoint['optimizer_G'])
    opt_d.load_state_dict(checkpoint['optimizer_D'])
    
    if 'scheduler_G' in checkpoint and scheduler_g is not None:
        scheduler_g.load_state_dict(checkpoint['scheduler_G'])
    if 'scheduler_D' in checkpoint and scheduler_d is not None:
        scheduler_d.load_state_dict(checkpoint['scheduler_D'])
        
    start_epoch = checkpoint['epoch'] + 1
    best_psnr = checkpoint.get('best_psnr', 0.0)
    return start_epoch, best_psnr

def train_geosafe_model(dataset, epochs=100, batch_size=4, learning_rate=2e-4, resume_path=None, device='cuda'):
    """
    V2 Training Engine: Alternating GAN loop with GeoSafe Loss and Checkpoint Resuming.
    """
    print(f"Initializing V2 Training Pipeline on {device}...")
    
    # 1. Train/Validation Split (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    # 2. Initialize Models (4x upscale)
    generator = Swin2SR(upscale=4).to(device)
    discriminator = UNetDiscriminator().to(device)
    
    # 3. Optimizers
    opt_g = optim.AdamW(generator.parameters(), lr=learning_rate, weight_decay=1e-4)
    opt_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 4. Learning Rate Schedulers (Cosine Annealing)
    sched_g = CosineAnnealingWarmRestarts(opt_g, T_0=10, T_mult=2, eta_min=1e-6)
    sched_d = CosineAnnealingWarmRestarts(opt_d, T_0=10, T_mult=2, eta_min=1e-6)
    
    # 5. GeoSafe Loss
    criterion = GeoSafeLoss(device=device)
    
    # 6. Mixed Precision Scaler
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    best_psnr = 0.0
    
    # 7. Resume Checkpoint (if provided)
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_psnr = load_checkpoint(resume_path, generator, discriminator, opt_g, opt_d, sched_g, sched_d)
    
    print(f"Starting Training from Epoch {start_epoch} to {epochs}...")
    
    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        g_loss_avg = 0.0
        d_loss_avg = 0.0
        
        for lr_batch, hr_batch in pbar:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            
            # ---------------------------
            # A. Train Discriminator
            # ---------------------------
            opt_d.zero_grad()
            with torch.cuda.amp.autocast():
                # Generate fake image (detach so gradients don't flow back to generator)
                fake_hr = generator(lr_batch)
                
                # Get D predictions
                d_real = discriminator(hr_batch)
                d_fake = discriminator(fake_hr.detach())
                
                # Calculate D Loss (RaGAN)
                d_loss = criterion.forward_d(d_real, d_fake)
                
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            
            # ---------------------------
            # B. Train Generator
            # ---------------------------
            opt_g.zero_grad()
            with torch.cuda.amp.autocast():
                # We already have fake_hr, but we need fresh gradients for generator
                fake_hr = generator(lr_batch)
                
                d_real = discriminator(hr_batch).detach() # D is fixed while training G
                d_fake = discriminator(fake_hr)
                
                # Calculate G Loss (L1 + Gradient + LPIPS + RaGAN)
                g_loss, loss_dict = criterion.forward_g(fake_hr, hr_batch, d_real, d_fake)
                
            scaler_g.scale(g_loss).backward()
            # Gradient clipping for generator stability
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            scaler_g.step(opt_g)
            scaler_g.update()
            
            g_loss_avg += g_loss.item()
            d_loss_avg += d_loss.item()
            pbar.set_postfix({"G_Loss": f"{g_loss.item():.3f}", "D_Loss": f"{d_loss.item():.3f}"})
            
        # Step schedulers
        sched_g.step()
        sched_d.step()
        
        # ---------------------------
        # C. Validation Loop
        # ---------------------------
        generator.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_val, hr_val in val_loader:
                lr_val, hr_val = lr_val.to(device), hr_val.to(device)
                
                # Force float32 for metrics calculation to ensure accuracy
                sr_val = generator(lr_val).clamp(0, 1)
                
                # We calculate PSNR on CPU using numpy (like the original metrics.py)
                val_psnr += compute_psnr(hr_val[0].cpu().numpy(), sr_val[0].cpu().numpy())
                
        val_psnr = val_psnr / len(val_loader)
        print(f"End of Epoch {epoch+1} | G_Loss: {g_loss_avg/len(train_loader):.3f} | Val PSNR: {val_psnr:.2f} dB")
        
        # ---------------------------
        # D. Checkpoint Saving
        # ---------------------------
        # Save regular checkpoint
        save_checkpoint(epoch, generator, discriminator, opt_g, opt_d, sched_g, sched_d, best_psnr, f"geosafe_checkpoint_epoch_{epoch+1}.pth")
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(generator.state_dict(), "geosafe_best_generator.pth")
            print(f"New Best PSNR ({best_psnr:.2f} dB)! Saved geosafe_best_generator.pth")

if __name__ == "__main__":
    print("V2 Train engine ready. Import train_geosafe_model in your Kaggle notebook to begin training.")
