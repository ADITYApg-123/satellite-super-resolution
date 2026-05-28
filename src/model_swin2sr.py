import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Allows the AI to learn which features (e.g., edges vs colors) are most important 
    at any given time by dynamically scaling the channels.
    """
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction, num_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    A Deep Residual Block with Channel Attention.
    This replaces the "fake" Swin blocks from V1.
    """
    def __init__(self, num_features=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.ca = ChannelAttention(num_features)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.ca(res)
        return x + res

class Swin2SR(nn.Module):
    """
    V2 Architecture: Deep Residual Attention Network (4x Upscale).
    We keep the class name 'Swin2SR' so we don't break the Streamlit app.
    """
    def __init__(self, upscale=4, num_features=64, num_blocks=8):
        super(Swin2SR, self).__init__()
        self.upscale = upscale
        
        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        
        # 2. Deep Feature Extraction (8 Residual Blocks)
        # This is where the AI learns complex textures like buildings and roads.
        self.deep_extraction = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        self.conv_after_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # 3. High-Quality Upsampling (4x)
        # 4x upscaling is done in two steps of 2x to prevent artifacting.
        if upscale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        elif upscale == 8:
            # Kept for backwards compatibility if needed
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * 64, kernel_size=3, padding=1),
                nn.PixelShuffle(8)
            )
        else:
            raise ValueError("Only upscale=4 and upscale=8 are supported.")
            
        # 4. Final Image Reconstruction
        self.conv_last = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Shallow features
        shallow_feat = self.conv_first(x)
        
        # Deep features + global residual
        deep_feat = self.deep_extraction(shallow_feat)
        deep_feat = self.conv_after_body(deep_feat)
        deep_feat = deep_feat + shallow_feat
        
        # Upscale and reconstruct
        hr_feat = self.upsample(deep_feat)
        out = self.conv_last(hr_feat)
        
        return out

def initialize_swin2sr_model(weight_path=None, upscale=4):
    """
    Initializes the V2 Super-Resolution model.
    """
    # Notice we now default to 4x upscaling!
    model = Swin2SR(upscale=upscale, num_features=64, num_blocks=8)
    
    if weight_path:
        try:
            state_dict = torch.load(weight_path, map_location='cpu')
            # Extract generator weights if loading from a GeoSafe checkpoint dict
            if 'generator' in state_dict:
                model.load_state_dict(state_dict['generator'], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weight_path}")
        except FileNotFoundError:
            print(f"Warning: Weights file not found at {weight_path}. Using random initialization.")
    
    model.eval()
    return model

def run_inference(model, input_tensor):
    """
    Runs a single forward pass without tracking gradients (saves VRAM).
    """
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
        
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    return output_tensor
