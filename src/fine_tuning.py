import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

# Project imports
from Dataset_Encoding import Imgur5KWordDataset, collate_fn, NUM_CLASSES, CHAR_TO_INDEX, INDEX_TO_CHAR, FIXED_HEIGHT
from model import CRNN

# --- Configs ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bs = 32
n_test = 200 # Set None for full test

# Paths
json_path = "./data/dataset_info/imgur5k_annotations_train.json"
img_root = "./data/imgur5k-dataset"
ckpt_base = "./checkpoints/crnn_pretrained.pth"
ckpt_ft = "./checkpoints/crnn_finetuned_mb.pth"

def get_edit_ops(ref, hyp):
    # Standard DP for Levenshtein with S/D/I tracking
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Init
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # Del
                           dp[i][j - 1] + 1,      # Ins
                           dp[i - 1][j - 1] + cost) # Sub
            
    # Backtrack for counts
    i, j = m, n
    cnt = {'S': 0, 'D': 0, 'I': 0}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            cnt['S'] += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            cnt['D'] += 1; i -= 1
        else:
            cnt['I'] += 1; j -= 1
            
    return cnt['S'], cnt['D'], cnt['I'], dp[m][n]

def decode(log_probs):
    # Greedy decode
    preds = log_probs.argmax(2).detach().cpu().numpy()
    res = []
    for seq in preds.T:
        txt = []
        prev = -1
        for idx in seq:
            if idx != 0 and idx != prev:
                txt.append(INDEX_TO_CHAR.get(idx, ""))
            prev = idx
        res.append("".join(txt))
    return res

def eval_model(name, path, loader):
    print(f">> Evaluating {name}...")
    
    # Load model
    model = CRNN(img_height=FIXED_HEIGHT, in_channels=1, num_output_classes=NUM_CLASSES).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()
    
    records = []
    with torch.no_grad():
        for imgs, labels, _, tgt_lens in loader:
            imgs = imgs.to(device)
            out = model(imgs) # [T, B, C]
            
            hyp_texts = decode(out)
            
            # Reconstruct GT texts
            ptr = 0
            for i, l in enumerate(tgt_lens):
                gt = "".join([INDEX_TO_CHAR[x.item()] for x in labels[ptr:ptr+l]])
                hyp = hyp_texts[i]
                ptr += l
                
                s, d, ins, dist = get_edit_ops(gt, hyp)
                records.append({
                    'Model': name,
                    'GT': gt, 'Hyp': hyp, 'Len': len(gt),
                    'Dist': dist, 'S': s, 'D': d, 'I': ins,
                    'Acc': gt == hyp
                })
                
    return pd.DataFrame(records)

# --- Main Script ---
# Setup Data
ds = Imgur5KWordDataset(json_path, img_root, FIXED_HEIGHT, CHAR_TO_INDEX)
# Use last 20%
idxs = list(range(int(len(ds) * 0.8), len(ds)))
if n_test: idxs = idxs[:n_test]

test_loader = DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=False, collate_fn=collate_fn)
print(f"Test samples: {len(idxs)}")

# Run Evals
df1 = eval_model("Baseline", ckpt_base, test_loader)
df2 = eval_model("Finetuned", ckpt_ft, test_loader)
df = pd.concat([df1, df2], ignore_index=True)

print("\nResults Preview:")
print(df.groupby('Model')[['Dist', 'Acc']].mean())

# --- Plotting ---
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Edit Distance Dist (Violin is better for density)
sns.violinplot(data=df[df.Dist <= 10], x="Dist", y="Model", ax=axes[0], palette="muted")
axes[0].set_title("Edit Distance Distribution")

# 2. Accuracy by Length bucket
df['LenBin'] = pd.cut(df['Len'], [0, 4, 8, 12, 100], labels=['Short', 'Mid', 'Long', 'XL'])
acc_stats = df.groupby(['Model', 'LenBin'], observed=True)['Acc'].mean().reset_index()
sns.barplot(data=acc_stats, x='LenBin', y='Acc', hue='Model', ax=axes[1], palette="viridis")
axes[1].set_title("Accuracy vs Word Length")
axes[1].set_ylabel("Exact Match Accuracy")

# 3. Error Types
errs = df.groupby('Model')[['S', 'D', 'I']].sum().reset_index().melt(id_vars='Model')
sns.barplot(data=errs, x='variable', y='value', hue='Model', ax=axes[2], palette="Set2")
axes[2].set_title("Error Type Breakdown")
axes[2].set_xlabel("Error Type (Sub/Del/Ins)")

plt.tight_layout()
plt.show()