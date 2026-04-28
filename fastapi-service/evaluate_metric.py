import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

from services.enhance_service import EnhanceService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOW_DIR  = os.path.join(BASE_DIR, "dataset/eval15/low")
HIGH_DIR = os.path.join(BASE_DIR, "dataset/eval15/high")
MODEL    = os.path.join(BASE_DIR, "models/alle_best_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

service = EnhanceService(lle_checkpoint=MODEL)

lpips_model = lpips.LPIPS(net='alex').to(device)
to_tensor   = transforms.ToTensor()

psnr_list  = []
ssim_list  = []
lpips_list = []

files = sorted(os.listdir(LOW_DIR))

for fname in tqdm(files):
    low_path  = os.path.join(LOW_DIR, fname)
    high_path = os.path.join(HIGH_DIR, fname)

    if not os.path.exists(high_path):
        continue

    low_img  = Image.open(low_path).convert("RGB")
    high_img = Image.open(high_path).convert("RGB")

    low_img  = low_img.resize((224, 224), Image.BICUBIC)
    high_img = high_img.resize((224, 224), Image.BICUBIC)

    result   = service.enhance(low_img)
    enhanced = result["image"]

    enhanced = enhanced.resize((224, 224), Image.BICUBIC)

    enh_np  = np.array(enhanced)
    high_np = np.array(high_img)


    psnr_list.append(psnr(high_np, enh_np, data_range=255))


    ssim_list.append(ssim(high_np, enh_np, channel_axis=2, data_range=255))


    t1 = (to_tensor(enhanced).unsqueeze(0).to(device) * 2) - 1
    t2 = (to_tensor(high_img).unsqueeze(0).to(device) * 2) - 1
    lpips_list.append(lpips_model(t1, t2).item())

print("\n" + "─"*60)
print(f"  AVERAGE  ({len(psnr_list)} images)")
print("─"*60)
print(f"  PSNR  = {np.mean(psnr_list):.3f} ↑")
print(f"  SSIM  = {np.mean(ssim_list):.4f} ↑")
print(f"  LPIPS = {np.mean(lpips_list):.4f} ↓")
print("─"*60)


print("\n" + "─"*60)
print("  COMPARISON WITH PAPER (LOL dataset)")
print("─"*60)
print(f"  {'Method':<25} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
print(f"  {'-'*52}")
print(f"  {'Paper ALL-E':<25} {'18.216':>8} {'0.763':>8} {'0.212':>8}")

print(f"  {'-'*52}")
print(f"  {'Your Model (epoch 200)':<25} {np.mean(psnr_list):>8.3f} {np.mean(ssim_list):>8.4f} {np.mean(lpips_list):>8.4f}")
print("─"*65)



