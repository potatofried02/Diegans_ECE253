import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Project imports
from Dataset_Encoding import Imgur5KWordDataset, collate_fn, NUM_CLASSES, CHAR_TO_INDEX, INDEX_TO_CHAR
from model import CRNN

# --- Configs ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_json = "./data/dataset_info/imgur5k_annotations_train.json"
val_json = "./data/dataset_info/imgur5k_annotations_val.json"
test_json = "./data/dataset_info/imgur5k_annotations_test.json"
img_root = "./data/imgur5k-dataset/"

# Hyperparams
H = 32
bs = 16
lr = 5e-6
epochs = 50
steps_per_epoch = 2000 # Limit batches per epoch
ckpt_dir = "./checkpoints"
resume_path = "./checkpoints/crnn_epoch_29_val_loss_1.6298.pth" # Set None to train from scratch

def decode(log_probs):
    # Greedy decode: argmax -> remove dups -> remove blank
    preds = log_probs.argmax(2).permute(1, 0).cpu().numpy()
    res = []
    for seq in preds:
        txt = []
        last = -1
        for idx in seq:
            if idx != 0 and idx != last:
                txt.append(INDEX_TO_CHAR.get(idx, ''))
            last = idx
        res.append("".join(txt))
    return res

def train_one_epoch(loader, model, criterion, opt, epoch):
    model.train()
    total_loss = 0
    
    # Iterate loader with a limit
    for i, (imgs, targets, ilens, tlens) in enumerate(loader):
        if i >= steps_per_epoch: 
            break
            
        imgs, targets = imgs.to(device), targets.to(device)
        
        opt.zero_grad()
        out = model(imgs) # [T, N, C]
        
        loss = criterion(out, targets, ilens, tlens)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Ep {epoch} | It {i}/{steps_per_epoch} | Loss: {loss.item():.4f}")

    return total_loss / steps_per_epoch

def validate(loader, model, criterion):
    model.eval()
    avg_loss = 0
    n_correct = 0
    n_total = 0
    
    with torch.no_grad():
        for imgs, targets, ilens, tlens in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            out = model(imgs)
            loss = criterion(out, targets, ilens, tlens)
            avg_loss += loss.item()
            
            # Calc Accuracy
            preds = decode(out)
            
            # Reconstruct GT strings
            ptr = 0
            for j, length in enumerate(tlens):
                gt_idxs = targets[ptr : ptr + length].tolist()
                gt = "".join([INDEX_TO_CHAR.get(x, '') for x in gt_idxs if x != 0])
                ptr += length
                
                if gt == preds[j]:
                    n_correct += 1
                n_total += 1
                
    return avg_loss / len(loader), (n_correct / n_total) * 100

if __name__ == "__main__":
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)

    # 1. Data
    # Common args
    dset_args = {'images_dir': img_root, 'target_height': H, 'char_to_index': CHAR_TO_INDEX}
    loader_args = {'batch_size': bs, 'num_workers': 4, 'collate_fn': collate_fn}
    
    train_loader = DataLoader(Imgur5KWordDataset(train_json, **dset_args), shuffle=True, **loader_args)
    val_loader = DataLoader(Imgur5KWordDataset(val_json, **dset_args), shuffle=False, **loader_args)
    test_loader = DataLoader(Imgur5KWordDataset(test_json, **dset_args), shuffle=False, **loader_args)

    # 2. Model & Opt
    model = CRNN(img_height=H, in_channels=1, num_output_classes=NUM_CLASSES, rnn_hidden_size=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=0, reduction='mean')

    # 3. Resume
    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        print(f">> Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Optional
        start_epoch = ckpt['epoch'] + 1
    else:
        print(">> Training from scratch")

    # 4. Loop
    best_acc = 0
    for epoch in range(start_epoch, epochs + 1):
        # Train
        t_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        
        # Val
        v_loss, v_acc = validate(val_loader, model, criterion)
        print(f"Ep {epoch} Result: Train Loss {t_loss:.4f} | Val Loss {v_loss:.4f} | Val Acc {v_acc:.2f}%")
        
        # Save
        save_name = f"{ckpt_dir}/crnn_ep{epoch}_loss{v_loss:.4f}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': v_loss,
            'val_acc': v_acc
        }, save_name)
        
    print("\n>> Final Test")
    t_loss, t_acc = validate(test_loader, model, criterion)
    print(f"Test Loss: {t_loss:.4f} | Test Acc: {t_acc:.2f}%")