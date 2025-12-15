import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from model import CRNN
from Dataset_Encoding import NUM_CLASSES, INDEX_TO_CHAR, FIXED_HEIGHT

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths - Keep it simple
ckpt1 = "./checkpoints/crnn_pretrained.pth" 
ckpt2 = "./checkpoints/crnn_finetuned_mb.pth"
img_path = "./demo/1.jpg"

def load_net(path):
    # Quick loader helper
    net = CRNN(FIXED_HEIGHT, 1, NUM_CLASSES, 256).to(device)
    print(f"Loading {path}...")
    chk = torch.load(path, map_location=device)
    # Handle dict vs state_dict automagically
    net.load_state_dict(chk['model_state_dict'] if 'model_state_dict' in chk else chk)
    net.eval()
    return net

def decode(logits):
    # Greedy decoder
    preds = torch.argmax(logits, dim=2).squeeze(1).cpu().numpy()
    res = []
    prev = -1
    for p in preds:
        if p != 0 and p != prev: # 0 is blank
            res.append(INDEX_TO_CHAR.get(p, ''))
        prev = p
    return "".join(res)

# 1. Load Models
net1 = load_net(ckpt1)
net2 = load_net(ckpt2)

# 2. Process Image (Script style, no function overhead)
raw_img = cv2.imread(img_path)
gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

h, w = gray.shape
scale = FIXED_HEIGHT / h
new_w = int(w * scale)

# Resize & Norm
img = cv2.resize(gray, (new_w, FIXED_HEIGHT))
img = img.astype(np.float32) / 255.0
x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device) # [B, C, H, W]

# 3. Inference
print(f"\nInference on: {img_path}")
with torch.no_grad():
    out1 = net1(x)
    pred1 = decode(out1)
    
    out2 = net2(x)
    pred2 = decode(out2)

# 4. Results
print("-" * 30)
print(f"crnn_pretrained.pth: {pred1}")
print(f"crnn_finetuned_mb.pth: {pred2}")
print("-" * 30)

# Quick viz
plt.figure(figsize=(10, 4))
plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
plt.title(f"A: {pred1}  |  B: {pred2}") # Put result in title, easier
plt.axis('off')
plt.show()