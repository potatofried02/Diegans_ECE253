import os
import math
import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage import io, color, img_as_float, img_as_ubyte
from tqdm import tqdm

# --- Config ---
src_dir = "./data/dataset/"
dst_dir = "./data/dataset_motion_blur/"

# Blur parameters
L = 15          # Kernel length
theta = 45      # Angle in degrees
sigma_n = 0.01  # Noise level (std dev)

# Setup
os.makedirs(dst_dir, exist_ok=True)

# 1. Create Motion Kernel (PSF h)
# Ensure odd size for proper centering
k_sz = L if L % 2 == 1 else L + 1
h = np.zeros((k_sz, k_sz))
cx, cy = k_sz // 2, k_sz // 2

# Calculate offsets
rad = math.radians(theta)
dx = int(np.cos(rad) * (L // 2))
dy = int(np.sin(rad) * (L // 2))

# Draw trajectory and normalize energy
cv2.line(h, (cx - dx, cy - dy), (cx + dx, cy + dy), 1, thickness=1)
h /= h.sum() 

# 2. Process Batch
files = sorted(os.listdir(src_dir))
print(f"Generating blur (L={L}, ang={theta}) for {len(files)} images...")

for f in tqdm(files):
    fpath = os.path.join(src_dir, f)
    
    # Load and prep
    raw = io.imread(fpath)
    if raw is None: continue 

    # Normalize [0,1] and grayscale
    im = img_as_float(raw)
    if im.ndim == 3:
        im = color.rgb2gray(im)

    # Convolve: g = f * h
    # boundary='symm' reduces ringing artifacts at edges
    im_blur = convolve2d(im, h, mode='same', boundary='symm')

    # Add noise
    if sigma_n > 0:
        np.random.seed(42 + len(f))
        noise = sigma_n * np.random.standard_normal(im_blur.shape)
        im_blur += noise

    # Save
    out_path = os.path.join(dst_dir, f)
    cv2.imwrite(out_path, img_as_ubyte(np.clip(im_blur, 0, 1)))

print("Done.")