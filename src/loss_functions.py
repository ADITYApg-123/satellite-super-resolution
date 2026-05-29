import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: A robust, differentiable variant of L1 loss.
    L1 loss is sharp but its derivative is undefined at zero, making it 
    unstable during the very final stages of convergence.
    """
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class GradientProfileLoss(nn.Module):
    """
    Gradient Profile Loss (Edge-Preserving Loss).
    Forces the AI to pay attention to high-frequency details (like building edges)
    using Sobel filters.
    """
    def __init__(self, device='cuda'):
        super(GradientProfileLoss, self).__init__()
        # Define Sobel edge detection kernels for X and Y axes
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(device)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(device)
        
        self.weight_x = nn.Parameter(kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(kernel_y, requires_grad=False)

    def forward(self, x, y):
        x_gray = torch.mean(x, dim=1, keepdim=True)
        y_gray = torch.mean(y, dim=1, keepdim=True)
        
        gx_x = F.conv2d(x_gray, self.weight_x, padding=1)
        gy_x = F.conv2d(x_gray, self.weight_y, padding=1)
        
        gx_y = F.conv2d(y_gray, self.weight_x, padding=1)
        gy_y = F.conv2d(y_gray, self.weight_y, padding=1)
        
        loss = torch.mean(torch.abs(gx_x - gx_y)) + torch.mean(torch.abs(gy_x - gy_y))
        return loss

# ==========================================
# PHASE 2 UPGRADES: GeoSafe Loss Components
# ==========================================

class UNetDiscriminator(nn.Module):
    """
    A lightweight UNet-style discriminator (inspired by Real-ESRGAN).
    Instead of outputting a single "real/fake" score for the whole image, 
    it outputs a per-pixel "real vs fake" map, giving the generator much 
    richer spatial feedback.
    """
    def __init__(self, in_channels=3, num_feat=64):
        super(UNetDiscriminator, self).__init__()
        # Downsample
        self.conv0 = nn.Conv2d(in_channels, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_feat * 4, num_feat * 8, kernel_size=4, stride=2, padding=1)
        
        # Upsample
        self.up1 = nn.ConvTranspose2d(num_feat * 8, num_feat * 4, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(num_feat * 4, num_feat * 2, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(num_feat * 2, num_feat, kernel_size=4, stride=2, padding=1)
        
        # Final classification
        self.conv_final = nn.Conv2d(num_feat, 1, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Very simple U-Net (no skip connections for speed, just an hourglass shape)
        feat0 = self.leaky_relu(self.conv0(x))
        feat1 = self.leaky_relu(self.conv1(feat0))
        feat2 = self.leaky_relu(self.conv2(feat1))
        feat3 = self.leaky_relu(self.conv3(feat2))
        
        up1 = self.leaky_relu(self.up1(feat3))
        up2 = self.leaky_relu(self.up2(up1))
        up3 = self.leaky_relu(self.up3(up2))
        
        out = self.conv_final(up3)
        return out

class RaGANLoss(nn.Module):
    """
    Relativistic Average GAN Loss.
    Standard GAN: "Is this image real or fake?"
    RaGAN: "Is this real image MORE REAL than the average fake image?"
    This provides significantly more stable gradients than vanilla BCE.
    """
    def __init__(self):
        super(RaGANLoss, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_pred, fake_pred):
        # D tries to make real > fake
        real_loss = self.bce_with_logits(real_pred - torch.mean(fake_pred), torch.ones_like(real_pred))
        fake_loss = self.bce_with_logits(fake_pred - torch.mean(real_pred), torch.zeros_like(fake_pred))
        return (real_loss + fake_loss) / 2

    def generator_loss(self, real_pred, fake_pred):
        # G tries to make fake > real
        real_loss = self.bce_with_logits(real_pred - torch.mean(fake_pred), torch.zeros_like(real_pred))
        fake_loss = self.bce_with_logits(fake_pred - torch.mean(real_pred), torch.ones_like(fake_pred))
        return (real_loss + fake_loss) / 2

class LPIPSWrapper(nn.Module):
    """
    Wraps the LPIPS library to compute perceptual texture similarity based on human judgments.
    Automatically normalizes input from [0, 1] to [-1, 1] as required by LPIPS.
    """
    def __init__(self, device='cuda'):
        super(LPIPSWrapper, self).__init__()
        if not LPIPS_AVAILABLE:
            raise ImportError("CRITICAL ERROR: 'lpips' library is not installed! You must run '!pip install lpips' in your Kaggle notebook before training.")
        self.lpips_model = lpips.LPIPS(net='vgg').to(device)
        # We freeze the VGG network
        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # LPIPS expects inputs in range [-1, 1]
        x_norm = (x * 2.0) - 1.0
        y_norm = (y * 2.0) - 1.0
        return self.lpips_model(x_norm, y_norm).mean()

class GeoSafeLoss(nn.Module):
    """
    The Master V2 Loss Function: L1 + LPIPS + Gradient + RaGAN.
    Designed specifically to prevent hallucinations in satellite imagery
    while maintaining razor-sharp edges and realistic textures.
    """
    def __init__(self, device='cuda', weight_l1=1.0, weight_lpips=0.04, weight_grad=0.30, weight_gan=0.005):
        super(GeoSafeLoss, self).__init__()
        self.weight_l1 = weight_l1
        self.weight_lpips = weight_lpips
        self.weight_grad = weight_grad
        self.weight_gan = weight_gan
        
        self.l1_loss = nn.L1Loss()
        self.gradient_loss = GradientProfileLoss(device=device)
        self.lpips_loss = LPIPSWrapper(device=device)
        self.ragan_loss = RaGANLoss()
        
    def forward_g(self, sr_pred, hr_true, d_pred_real, d_pred_fake):
        """ Computes the total loss for the Generator """
        # Physics anchor
        l_l1 = self.l1_loss(sr_pred, hr_true) * self.weight_l1
        
        # Edge penalty
        l_grad = self.gradient_loss(sr_pred, hr_true) * self.weight_grad
        
        # Perceptual texture
        l_lpips = self.lpips_loss(sr_pred, hr_true) * self.weight_lpips
        
        # GAN crispness
        l_gan = self.ragan_loss.generator_loss(d_pred_real, d_pred_fake) * self.weight_gan
        
        total_g_loss = l_l1 + l_grad + l_lpips + l_gan
        return total_g_loss, {'l1': l_l1.item(), 'grad': l_grad.item(), 'lpips': l_lpips.item(), 'gan': l_gan.item()}
        
    def forward_d(self, d_pred_real, d_pred_fake):
        """ Computes the loss for the Discriminator """
        return self.ragan_loss.discriminator_loss(d_pred_real, d_pred_fake)
