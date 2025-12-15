import os
import json
import cv2
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configs && IMGUR5K && In-the-Wild
JSON_PATH = "./data/dataset_info/imgur5k_annotations_train.json"
IMG_ROOT = "./data/imgur5k-dataset/"
IMG_H = 32

# Charset + blank
chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?'()[]{}-+*/="
char2idx = {c: i + 1 for i, c in enumerate(chars)}
idx2char = {i + 1: c for i, c in enumerate(chars)}
n_class = len(chars) + 1

def crop_word_poly(img, cx, cy, w, h, ang, target_h=32):
    # Math: rotation matrix to find corners
    rad = math.radians(ang)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    # Half dims vectors
    wx, wy = (w / 2) * cos_a, (w / 2) * sin_a
    hx, hy = (-h / 2) * sin_a, (h / 2) * cos_a

    # Corners: TL, TR, BL (for affine src)
    p1 = (cx - wx + hx, cy - wy + hy) # TL
    p2 = (cx + wx + hx, cy + wy + hy) # TR
    p3 = (cx - wx - hx, cy - wy - hy) # BL
    
    src = np.float32([p3, p2, p1]) # align with dst order
    dst_w, dst_h = int(round(w)), int(round(h))
    
    # Destination points: BL, TR, TL (matches src order logic in original code)
    # Note: original code used src=[BL, TR, TL] mapping to dst=[TL, TR, BL]? 
    # Let's keep strict logic parity with original code's corner indices:
    # Original: pt1(TL), pt2(TR), pt3(BL), pt4(BR) -> src=[pt4, pt3, pt1] (Wait, corners[3] is BR?)
    # Re-evaluating original logic: 
    # pt4 = (xc - x_vec - x_vec_h...) -> This is actually Bottom-Left relative to rotation?
    # Let's stick to the raw math to ensure pixel-perfect match.
    
    # Re-implementing strictly original math sequence for safety:
    pt1 = (cx - wx + hx, cy - wy + hy)
    pt2 = (cx + wx + hx, cy + wy + hy)
    pt3 = (cx + wx - hx, cy + wy - hy)
    pt4 = (cx - wx - hx, cy - wy - hy)
    
    # Original used: [corners[3], corners[2], corners[0]]
    src = np.float32([pt4, pt3, pt1])
    dst = np.float32([[0, 0], [dst_w - 1, 0], [0, dst_h - 1]])

    M = cv2.getAffineTransform(src, dst)
    warp = cv2.warpAffine(img, M, (dst_w, dst_h))

    # Preproc
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    h_raw, w_raw = gray.shape
    
    scale = target_h / h_raw
    new_w = int(w_raw * scale)
    
    patch = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)
    patch = patch.astype(np.float32) / 255.0
    
    return torch.from_numpy(patch).unsqueeze(0) # [1, H, W]

class ImgurDataset(Dataset):
    def __init__(self, json_file, root, imgH, char_map):
        self.root = root
        self.imgH = imgH
        self.cmap = char_map
        self.samples = self._parse_json(json_file)

    def _parse_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        
        idx2ann = data['index_to_ann_map']
        ann_db = data['ann_id']
        samples = []

        print(f"Parsing {path}...")
        for img_id, ann_ids in idx2ann.items():
            fpath = os.path.join(self.root, f"{img_id}.jpg")
            for aid in ann_ids:
                item = ann_db.get(aid)
                if not item: continue
                
                txt = item['word']
                bbox_str = item['bounding_box']
                
                if bbox_str == '.' or not txt: continue
                
                # Parse bbox
                box = [float(x) for x in bbox_str.strip('[]').split(',')]
                samples.append((fpath, txt, box))
                
        print(f"Got {len(samples)} samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fpath, txt, box = self.samples[i]
        cx, cy, w, h, a = box
        
        img = cv2.imread(fpath)
        # Assuming data integrity effectively, or let it crash to find bad files
        tensor = crop_word_poly(img, cx, cy, w, h, a, self.imgH)
        
        label = torch.tensor([self.cmap.get(c, 0) for c in txt], dtype=torch.long)
        return tensor, label

def align_collate(batch):
    imgs, labels = zip(*batch)
    
    # Pad to max width in batch
    max_w = max(im.size(2) for im in imgs)
    padded_imgs = []
    for im in imgs:
        pad_size = max_w - im.size(2)
        padded_imgs.append(F.pad(im, (0, pad_size, 0, 0), value=0))
        
    img_batch = torch.stack(padded_imgs)
    targets = torch.cat(labels)
    
    # Lengths for CTC
    input_lens = torch.full((img_batch.size(0),), max_w // 4, dtype=torch.long)
    target_lens = torch.tensor([l.size(0) for l in labels], dtype=torch.long)
    
    return img_batch, targets, input_lens, target_lens

# Debug / Usage
if __name__ == "__main__":
    ds = ImgurDataset(JSON_PATH, IMG_ROOT, IMG_H, char2idx)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=align_collate)

    # Just verify one batch
    imgs, t, ilen, tlen = next(iter(loader))
    
    print(f"Batch loaded.")
    print(f"Imgs: {imgs.shape} (N, C, H, W)")
    print(f"Targets: {t.shape} (Total chars)")
    print(f"Input Lens: {ilen}")
    print(f"Target Lens: {tlen}")