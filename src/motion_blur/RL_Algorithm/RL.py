import os
import numpy as np
import cv2
from skimage import color, restoration, io, img_as_float, img_as_ubyte
from scipy.signal import convolve2d as conv2
from tqdm import tqdm

# --- Config ---
src_dir = "./data/dataset_motion_blur/"
dst_dir = "./data/dataset_RL/"
sim_blur = True  # True: synth blur -> fix; False: real blur -> fix

# Setup output
os.makedirs(dst_dir, exist_ok=True)

# Define PSF (15x15 box blur as per report)
# Keeping this simple; real motion blur would likely use a linear kernel.
k_size = 15
psf = np.ones((k_size, k_size)) / (k_size**2)

files = sorted(os.listdir(src_dir))
print(f"Processing {len(files)} images | Mode: {'Simulate' if sim_blur else 'Direct'}")

for fname in tqdm(files):
    # Load and prep
    fpath = os.path.join(src_dir, fname)
    raw = io.imread(fpath)
    
    # Standardize to float [0,1] gray
    im = img_as_float(raw)
    if im.ndim == 3:
        im = color.rgb2gray(im)
    
    # 1. Simulate Blur (if needed)
    if sim_blur:
        # 'symm' boundary to avoid black borders during convolution
        im = conv2(im, psf, 'same', boundary='symm')
        
        # Add slight noise
        np.random.seed(0)
        noise = 0.05 * im.std() * np.random.standard_normal(im.shape)
        im = np.clip(im + noise, 0, 1)

    # 2. RL Deconvolution setup
    # Reflection padding to stop ringing at edges
    rad = psf.shape[0] // 2
    pad_w = rad * 2 
    im_pad = np.pad(im, pad_w, mode='reflect')
    
    # 3. Run Restoration
    # 30 iterations gave best results in testing
    res_pad = restoration.richardson_lucy(im_pad, psf, num_iter=30, clip=False)
    
    # 4. Crop padding
    h, w = im.shape
    res = res_pad[pad_w : h + pad_w, pad_w : w + pad_w]
    
    # Save result
    out_path = os.path.join(dst_dir, fname)
    cv2.imwrite(out_path, img_as_ubyte(np.clip(res, 0, 1)))

print("Done.")