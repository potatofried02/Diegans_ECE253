import torch
import matplotlib.pyplot as plt
import math
import os
import numpy as np
from torch.utils.data import DataLoader, Subset

# Project imports
from Dataset_Encoding import Imgur5KWordDataset, collate_fn, NUM_CLASSES, CHAR_TO_INDEX, INDEX_TO_CHAR, FIXED_HEIGHT
from model import CRNN

# --- Configs ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "./checkpoints/crnn_pretrained.pth"
json_path = "./data/dataset_info/imgur5k_annotations_test.json"
img_dir = "./data/imgur5k-dataset"

# Range to visualize
START, END = 500, 520

# Simple Lev dist (if you don't have 'pip install editdistance')
def calc_cer(r, h):
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]: d[i][j] = d[i-1][j-1]
            else: d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    return d[len(r)][len(h)] / len(r) if len(r) > 0 else 1.0

def decode(log_probs):
    # Greedy decode
    preds = log_probs.argmax(2).cpu().numpy()
    res = []
    for seq in preds.T: # [T, B] -> iterate batch
        txt = []
        prev = -1
        for idx in seq:
            if idx != 0 and idx != prev:
                txt.append(INDEX_TO_CHAR.get(idx, ''))
            prev = idx
        res.append("".join(txt))
    return res

def run_viz():
    # 1. Load Data
    ds = Imgur5KWordDataset(json_path, img_dir, FIXED_HEIGHT, CHAR_TO_INDEX)
    # Just slice the indices we care about
    indices = list(range(START, min(END, len(ds))))
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset loaded. Visualizing {len(indices)} samples.")

    # 2. Load Model
    model = CRNN(img_height=FIXED_HEIGHT, in_channels=1, num_output_classes=NUM_CLASSES).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()

    # 3. Setup Plot
    n = len(indices)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2.5))
    axes = axes.flatten() if n > 1 else [axes]

    # 4. Inference loop
    total_cer = 0
    total_len = 0
    correct = 0

    with torch.no_grad():
        for i, (img, target, _, tgt_len) in enumerate(loader):
            img = img.to(device)
            out = model(img) # [T, B, C]
            
            # Decode
            pred = decode(out)[0]
            
            # Recover GT
            gt_idxs = target[:tgt_len[0]].tolist()
            gt = "".join([INDEX_TO_CHAR.get(x, '') for x in gt_idxs])

            # Metrics
            cer = calc_cer(gt, pred)
            total_cer += cer * len(gt)
            total_len += len(gt)
            if gt == pred: correct += 1

            # Visualize
            ax = axes[i]
            # Show image (CHW -> HWC)
            ax.imshow(img.cpu().squeeze().numpy(), cmap='gray')
            
            color = 'green' if gt == pred else 'red'
            ax.set_title(f"GT: {gt}\nPred: {pred}\nCER: {cer:.2f}", color=color, fontsize=10)
            ax.axis('off')

    # Cleanup empty plots
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])

    print(f"\nResults for range [{START}:{END}]:")
    print(f"WAcc: {correct/n*100:.2f}%")
    print(f"Avg CER: {total_cer/total_len:.4f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_viz()