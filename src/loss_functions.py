import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: A robust, differentiable variant of L1 loss.
    L1 loss is sharp but its derivative is undefined at zero, making it 
    unstable during the very final stages of convergence.
    Charbonnier adds a tiny epsilon to smooth the gradient near zero,
    preventing the model from endlessly oscillating around the perfect answer.
    """
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # sqrt(diff^2 + eps^2)
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class GradientProfileLoss(nn.Module):
    """
    Gradient Profile Loss (Edge-Preserving Loss).
    This forces the AI to pay attention to high-frequency details (like building edges).
    It works by running a 'Sobel Filter' (an edge-detection algorithm) over both
    the AI's prediction and the true High-Res image, and penalizing differences
    in the edge maps.
    """
    def __init__(self, device='cuda'):
        super(GradientProfileLoss, self).__init__()
        # Define Sobel edge detection kernels for X and Y axes
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(device)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(device)
        
        # We use nn.Parameter with requires_grad=False so they act as static filters
        self.weight_x = nn.Parameter(kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(kernel_y, requires_grad=False)

    def forward(self, x, y):
        # We only apply the edge detector to the luminance (brightness) channel, 
        # so we average across the RGB channels first.
        x_gray = torch.mean(x, dim=1, keepdim=True)
        y_gray = torch.mean(y, dim=1, keepdim=True)
        
        # Calculate gradients (edges)
        gx_x = F.conv2d(x_gray, self.weight_x, padding=1)
        gy_x = F.conv2d(x_gray, self.weight_y, padding=1)
        
        gx_y = F.conv2d(y_gray, self.weight_x, padding=1)
        gy_y = F.conv2d(y_gray, self.weight_y, padding=1)
        
        # Mean absolute error of the edges
        loss = torch.mean(torch.abs(gx_x - gx_y)) + torch.mean(torch.abs(gy_x - gy_y))
        return loss

class CompositeHallucinationLoss(nn.Module):
    """
    Our Master Loss Function.
    Combines pixel accuracy (Charbonnier) with edge sharpness (Gradient Profile).
    You could also inject SSIM here!
    """
    def __init__(self, alpha=1.0, beta=0.1, device='cuda'):
        super(CompositeHallucinationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.charbonnier = CharbonnierLoss()
        self.gradient_loss = GradientProfileLoss(device=device)
        
    def forward(self, sr_pred, hr_true):
        # 1. Base color and structure loss
        l_charb = self.charbonnier(sr_pred, hr_true)
        
        # 2. Edge/Sharpness loss
        l_grad = self.gradient_loss(sr_pred, hr_true)
        
        # Combine them with weighting factors
        total_loss = (self.alpha * l_charb) + (self.beta * l_grad)
        return total_loss
