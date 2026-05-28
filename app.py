import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison
import torch
import sys
import os
import requests

# Append src to path to import our modules
sys.path.append(os.path.abspath("src"))
from metrics import compute_hallucination_score, compute_psnr, compute_ssim
from model_swin2sr import initialize_swin2sr_model, run_inference

# Check for Real-ESRGAN
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REALESRGAN = True
except ImportError:
    HAS_REALESRGAN = False

st.set_page_config(page_title="Satellite Super-Resolution", layout="wide")

st.title("🌍 Satellite Imagery Super-Resolution")
st.markdown("### Upgrade low-resolution satellite data to commercial-grade sharpness.")

# --- UI Controls ---
with st.sidebar:
    st.header("Control Panel")
    
    model_choice = st.selectbox("Select AI Model", [
        "Swin2SR v2 (GeoSafe - 4x)",
        "Swin2SR v1 (L1 Only - 8x)",
        "Real-ESRGAN (Pretrained - 4x)"
    ])
    
    st.markdown("---")
    input_mode = st.radio("Input Method", ["File Upload", "Live Earth Engine"])
    
    uploaded_file = None
    gee_lat = 0.0
    gee_lon = 0.0
    
    if input_mode == "File Upload":
        st.markdown("Upload a raw Sentinel-2 patch to run inference.")
        uploaded_file = st.file_uploader("Upload .tif or .png", type=["png", "jpg", "jpeg", "tif"])
    else:
        st.markdown("Stream live satellite data from Google Earth Engine.")
        gee_project = st.text_input("Earth Engine Project ID", value="", help="Required by Google.")
        gee_lat = st.number_input("Latitude", value=25.1200, format="%.4f") 
        gee_lon = st.number_input("Longitude", value=55.1200, format="%.4f")
        
    alpha_blend = st.slider("Ensemble Blend (AI vs Bicubic)", 0.0, 1.0, 1.0)
    run_btn = st.button("Run Super-Resolution")

# --- Main App Logic ---
lr_pil = None

if input_mode == "File Upload" and uploaded_file is not None:
    lr_pil = Image.open(uploaded_file).convert('RGB')

elif input_mode == "Live Earth Engine" and run_btn:
    with st.spinner("Connecting to Google Earth Engine..."):
        try:
            import ee
            from io import BytesIO
            
            try:
                if gee_project.strip():
                    ee.Initialize(project=gee_project.strip())
                else:
                    ee.Initialize()
            except Exception as e:
                st.error(f"Earth Engine Initialization failed! {str(e)}")
                st.stop()
                
            delta = 0.0075
            roi = ee.Geometry.Rectangle([gee_lon - delta, gee_lat - delta, gee_lon + delta, gee_lat + delta])
            
            from data_loader import fetch_gee_image_with_retry
            gee_image = fetch_gee_image_with_retry(roi, '2023-01-01', '2023-12-31')
            
            url = gee_image.getThumbURL({
                'dimensions': 512,
                'format': 'png',
                'min': 0,
                'max': 2000
            })
            response = requests.get(url)
            lr_pil = Image.open(BytesIO(response.content)).convert('RGB')
            st.success("Successfully streamed live patch from Earth Engine!")
            
        except Exception as e:
            st.error(f"Failed to fetch from Earth Engine: {e}")
            st.stop()

if lr_pil is not None:
    
    # 1. Determine Model Configuration
    if model_choice == "Swin2SR v1 (L1 Only - 8x)":
        scale_factor = 8
        weights_path = "swin2sr_epoch_10.pth"
    elif model_choice == "Swin2SR v2 (GeoSafe - 4x)":
        scale_factor = 4
        weights_path = "geosafe_best_generator.pth"
    else:
        scale_factor = 4
        weights_path = "RealESRGAN_x4plus.pth"
        
    # 2. Prevent RAM Explosions
    MAX_INPUT_SIZE = 512 if scale_factor == 4 else 256
    original_size = lr_pil.size
    if max(original_size) > MAX_INPUT_SIZE:
        ratio = MAX_INPUT_SIZE / max(original_size)
        new_w = int(original_size[0] * ratio)
        new_h = int(original_size[1] * ratio)
        lr_pil = lr_pil.resize((new_w, new_h), Image.LANCZOS)
        st.info(f"📐 Image resized to {new_w}×{new_h} to fit in laptop RAM. AI Output will be {new_w*scale_factor}×{new_h*scale_factor}px.")
        
    lr_width, lr_height = lr_pil.size
    hr_width, hr_height = lr_width * scale_factor, lr_height * scale_factor
    blurry_hr = lr_pil.resize((hr_width, hr_height), Image.BICUBIC)

    # 3. Model Loading Logic
    @st.cache_resource
    def load_swin_model(path, upscale):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = initialize_swin2sr_model(weight_path=path, upscale=upscale)
        return model.to(device), device
        
    @st.cache_resource
    def load_realesrgan_model():
        if not HAS_REALESRGAN:
            return None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        # Download weights automatically if missing
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
        if not os.path.exists(model_path):
            os.makedirs('weights', exist_ok=True)
            response = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
        upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=True, device=device)
        return upsampler

    # 4. Run Inference
    with st.spinner(f"Running {model_choice}..."):
        if "Real-ESRGAN" in model_choice:
            if not HAS_REALESRGAN:
                st.error("Real-ESRGAN requires the `realesrgan` and `basicsr` python packages. Run `pip install realesrgan basicsr` to use this feature.")
                st.stop()
            upsampler = load_realesrgan_model()
            img_np = np.array(lr_pil)
            sharp_np, _ = upsampler.enhance(img_np, outscale=4)
            sharp_hr = Image.fromarray(sharp_np)
        else:
            if not os.path.exists(weights_path):
                st.error(f"❌ Missing weights: `{weights_path}`. Please train the model or download the checkpoint to the project root.")
                st.stop()
            model, device = load_swin_model(weights_path, scale_factor)
            lr_tensor = torch.from_numpy(np.array(lr_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            lr_tensor = lr_tensor.to(device)
            sr_tensor = run_inference(model, lr_tensor)
            sr_tensor_cpu = sr_tensor.squeeze(0).cpu().clamp(0, 1)
            sharp_np = (sr_tensor_cpu.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            sharp_hr = Image.fromarray(sharp_np)

    # 5. Ensemble Blend
    blurry_np = np.array(blurry_hr, dtype=np.float32)
    sharp_np_float = np.array(sharp_hr, dtype=np.float32)
    blended_np = (alpha_blend * sharp_np_float) + ((1.0 - alpha_blend) * blurry_np)
    blended_pil = Image.fromarray(blended_np.astype(np.uint8))
    
    # --- Interactive Visualizer ---
    st.markdown("---")
    st.subheader("Interactive Comparison")
    
    image_comparison(
        img1=blended_pil,
        img2=blurry_hr,
        label1=f"AI Output ({alpha_blend*100:.0f}% Blend)",
        label2="Baseline (Bicubic)",
        width=1200,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
    )
    
    # --- Real Metrics Calculation ---
    st.markdown("---")
    st.subheader("Safety & Evaluation Metrics")
    
    # To calculate real PSNR/SSIM, we downscale the AI output back to the original LR size
    # and compare it mathematically to the true original input (consistency check).
    ai_downscaled_np = np.array(blended_pil.resize((lr_width, lr_height), Image.BICUBIC), dtype=np.float32) / 255.0
    original_lr_np = np.array(lr_pil, dtype=np.float32) / 255.0
    
    real_psnr = compute_psnr(original_lr_np, ai_downscaled_np)
    real_ssim = compute_ssim(original_lr_np, ai_downscaled_np)
    
    # Hallucination Score
    mock_sr_tensor = torch.from_numpy(blended_np).permute(2, 0, 1).unsqueeze(0) / 255.0
    mock_lr_tensor = torch.from_numpy(original_lr_np * 255.0).permute(2, 0, 1).unsqueeze(0) / 255.0
    hallucination_score = compute_hallucination_score(mock_sr_tensor, mock_lr_tensor, scale_factor=scale_factor)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Hallucination Score", f"{hallucination_score:.4f}", "-Lower is safer", delta_color="inverse")
    col2.metric("Consistency PSNR", f"{real_psnr:.2f} dB", help="How perfectly the AI image matches the original when zoomed back out.")
    col3.metric("Consistency SSIM", f"{real_ssim:.4f}", help="1.0 is a perfect structural match.")

else:
    st.info("👈 Please select a model and upload an image in the sidebar to begin.")
