import os
from PIL import Image

import torch
import torchvision.transforms as T

from shrnet_model import SimpleSHRNet

base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")

glare_dir = os.path.join(data_dir, "imgur5k_glare")
out_dir   = os.path.join(data_dir, "imgur5k_shrnet")
os.makedirs(out_dir, exist_ok=True)

model_path = os.path.join(models_dir, "shrnet.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = SimpleSHRNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loading complete:", model_path)

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])

files = [f for f in os.listdir(glare_dir)
         if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]]

print("Number of glare images to process:", len(files))

for fname in sorted(files):
    in_path = os.path.join(glare_dir, fname)
    print(f"SHRNET Processing: {fname}")

    img = Image.open(in_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)

    y_img = y.squeeze(0).cpu()
    y_img = T.ToPILImage()(y_img)

    name, ext = os.path.splitext(fname)   
    key = name.replace("_glare", "")      
    out_name = f"{key}_shrnet.png"
    out_path = os.path.join(out_dir, out_name)
    y_img.save(out_path)

print("Generate of SHR-Net results completed for all glare images.")
