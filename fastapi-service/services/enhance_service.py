import io
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.policy_network import PolicyNetwork, apply_pac


# =========================
# CONFIG
# =========================
N_STEPS_LLE = 6
IMAGE_SIZE = 224

_to_tensor = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

_to_pil = transforms.ToPILImage()


# =========================
# 🔥 HIGH-QUALITY DENOISING (LIKE 4th IMAGE)
# =========================
def fast_post_process(pil_image):
    img = np.array(pil_image).astype(np.uint8)

    # STEP 1: Denoise the image first
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 4, 4, 7, 21)

    # STEP 2: Extract detail from ORIGINAL (has detail + noise)
    # Extract detail from DENOISED (has detail, no noise)
    detail = cv2.subtract(img, denoised)  # this is noise + detail
    
    # Separate real edges from noise using Laplacian magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge_mask = cv2.Laplacian(gray, cv2.CV_64F)
    edge_mask = np.abs(edge_mask)
    edge_mask = (edge_mask / edge_mask.max() * 255).astype(np.uint8)
    edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)  # smooth mask
    edge_mask = cv2.threshold(edge_mask, 15, 255, cv2.THRESH_BINARY)[1]
    # Dilate to cover full edges
    edge_mask = cv2.dilate(edge_mask, np.ones((2, 2), np.uint8), iterations=1)
    edge_mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask])

    # STEP 3: Sharpen ONLY on edge areas, keep denoised on flat areas
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 0.8)
    sharpened = cv2.addWeighted(denoised, 1.6, gaussian, -0.6, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Blend: sharpened where edges, denoised where flat/noise
    img = np.where(edge_mask_3ch == 255, sharpened, denoised)
    img = img.astype(np.uint8)

    # STEP 4: Final very light cleanup
    img = cv2.bilateralFilter(img, 3, 8, 8)

    return Image.fromarray(img)
# SERVICE CLASS
# =========================
class EnhanceService:

    def __init__(self, lle_checkpoint=None, device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        print(f"Using device: {self.device}")

        self.policy_lle = PolicyNetwork().to(self.device)

        if lle_checkpoint and os.path.exists(lle_checkpoint):
            try:
                state = torch.load(lle_checkpoint, map_location=self.device)

                if isinstance(state, dict) and "policy_net" in state:
                    self.policy_lle.load_state_dict(state["policy_net"])
                else:
                    self.policy_lle.load_state_dict(state)

                print(f"✅ Loaded model: {lle_checkpoint}")

            except Exception as e:
                print(f"❌ Load failed: {e}")

        self.policy_lle.eval()

    def _tensor_to_pil(self, tensor, size):
        tensor = tensor.squeeze(0).cpu().clamp(0, 1)
        pil = _to_pil(tensor)
        return pil.resize(size, Image.LANCZOS)

    def enhance(self, pil_image):
        orig_size = pil_image.size

        tensor = _to_tensor(pil_image).unsqueeze(0).to(self.device)

        # 🔹 LLE
        with torch.no_grad():
            for _ in range(N_STEPS_LLE):
                action, _ = self.policy_lle.select_action(tensor)
                tensor = apply_pac(tensor, action)

        lle_img = self._tensor_to_pil(tensor, orig_size)

        # 🔹 DENOISE
        denoised_img = fast_post_process(lle_img)

        return lle_img, denoised_img

    def enhance_to_bytes(self, pil_image, fmt="PNG"):
        lle_img, denoised_img = self.enhance(pil_image)

        # ✅ LLE → BytesIO
        lle_buf = io.BytesIO()
        lle_img.save(lle_buf, format=fmt)
        lle_buf.seek(0)

        # ✅ DENOISED → BytesIO
        denoise_buf = io.BytesIO()
        denoised_img.save(denoise_buf, format=fmt)
        denoise_buf.seek(0)

        return lle_buf, denoise_buf

