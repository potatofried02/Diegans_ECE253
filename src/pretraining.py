import os
import json
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Paths & Configs
JSON_PATH = "./data/dataset_info/imgur5k_annotations_train.json"
IMG_ROOT = "./data/imgur5k-dataset/"
IMG_H = 32

def get_corners(cx, cy, w, h, ang):
    # Simple rotation logic
    rad = math.radians(ang)
    cos, sin = math.cos(rad), math.sin(rad)
    
    # Vectors for half-width and half-height
    wx, wy = (w / 2) * cos, (w / 2) * sin
    hx, hy = (-h / 2) * sin, (h / 2) * cos
    
    # Calculate 4 corners: BL, BR, TR, TL (matching original logic)
    # (Using list comprehension or explicit definition for clarity)
    pts = [
        (cx - wx + hx, cy - wy + hy), # BL
        (cx + wx + hx, cy + wy + hy), # BR
        (cx + wx - hx, cy + wy - hy), # TR
        (cx - wx - hx, cy - wy - hy)  # TL
    ]
    return np.float32(pts)

def crop_patch(img, cx, cy, w, h, ang, target_h):
    # Need 3 points for Affine: TL, TR, BL
    corners = get_corners(cx, cy, w, h, ang)
    src = np.float32([corners[3], corners[2], corners[0]]) # TL, TR, BL
    
    dst_w, dst_h = int(w), int(h)
    dst = np.float32([[0, 0], [dst_w, 0], [0, dst_h]])
    
    # Warp
    M = cv2.getAffineTransform(src, dst)
    warp = cv2.warpAffine(img, M, (dst_w, dst_h))
    
    # Post-process: Gray -> Resize -> Norm
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    scale = target_h / gray.shape[0]
    new_w = int(gray.shape[1] * scale)
    
    patch = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return patch / 255.0

def viz_sample():
    # Load data
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    idx2ann = data['index_to_ann_map']
    ann_db = data['ann_id']
    
    # Pick random valid image
    valid_idxs = list(idx2ann.keys())
    idx = random.choice(valid_idxs)
    fname = f"{idx}.jpg"
    fpath = os.path.join(IMG_ROOT, fname)
    
    print(f"Processing: {fname}")
    
    img = cv2.imread(fpath)
    if img is None:
        print(f"Failed to load {fpath}")
        return

    vis_img = img.copy()
    crops = []
    
    # Process annotations
    for aid in idx2ann[idx]:
        ann = ann_db.get(aid)
        if not ann or ann['bounding_box'] == '.': continue
        
        # Parse box
        box_str = ann['bounding_box']
        try:
            cx, cy, w, h, a = [float(x) for x in box_str.strip('[]').split(',')]
        except: continue
            
        # 1. Draw box
        pts = get_corners(cx, cy, w, h, a).astype(np.int32)
        cv2.polylines(vis_img, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        
        # 2. Crop 
        patch = crop_patch(img, cx, cy, w, h, a, IMG_H)
        crops.append((ann['word'], patch))

    # Visualize
    if not crops: 
        print("No valid crops found.")
        return

    n = len(crops) + 1
    cols = 3
    rows = math.ceil(n / cols)
    
    plt.figure(figsize=(12, 4 * rows))
    
    # Original image
    plt.subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Source: {fname}")
    plt.axis('off')
    
    # Crops
    for i, (txt, patch) in enumerate(crops):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(patch, cmap='gray')
        plt.title(f"{txt}\n{patch.shape}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    viz_sample()