import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image


N_ACTIONS_D = 9
N_STEPS_D   = 5
GAMMA_D     = 0.9
W1          = 1.0
WD          = 0.5

ACTION_GAUSSIAN_S1  = 0
ACTION_GAUSSIAN_S2  = 1
ACTION_MEDIAN       = 2
ACTION_BOX          = 3
ACTION_BILATERAL_S1 = 4
ACTION_BILATERAL_S2 = 5
ACTION_PIX_INC      = 6
ACTION_PIX_DEC      = 7
ACTION_DO_NOTHING   = 8


def apply_action(img_np, a):

    if a == ACTION_GAUSSIAN_S1:  return cv2.GaussianBlur(img_np, (3, 3), 1.0)
    if a == ACTION_GAUSSIAN_S2:  return cv2.GaussianBlur(img_np, (5, 5), 2.0)
    if a == ACTION_MEDIAN:       return cv2.medianBlur(img_np, 3)
    if a == ACTION_BOX:          return cv2.boxFilter(img_np, -1, (3, 3))
    if a == ACTION_BILATERAL_S1: return cv2.bilateralFilter(img_np, 5, 10, 15)
    if a == ACTION_BILATERAL_S2: return cv2.bilateralFilter(img_np, 9, 25, 15)
    if a == ACTION_PIX_INC:      return np.clip(img_np + 1, 0, 255).astype(np.uint8)
    if a == ACTION_PIX_DEC:      return np.clip(img_np - 1, 0, 255).astype(np.uint8)
    return img_np  


def apply_action_map(img_tensor, action_map):
    
    img_np     = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    action_np  = action_map.squeeze(0).cpu().numpy()

    out = img_np.copy()
    for a in range(N_ACTIONS_D):
        mask = (action_np == a)
        if mask.any():
            filtered = apply_action(img_np, a)
            out[mask] = filtered[mask]

    out_tensor = torch.from_numpy(out.astype(np.float32) / 255.0)
    return out_tensor.permute(2, 0, 1).unsqueeze(0)  


class DenoiseValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   

    def compute_gradient(self, state, Rt):
        V    = self.forward(state)
        loss = F.mse_loss(V, Rt.detach())
        return loss, V


class DenoisePolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, N_ACTIONS_D, 3, padding=1)
        )

    def forward(self, x):
        
        return F.softmax(self.net(x), dim=1)   

    def select_action(self, state):
        
        with torch.no_grad():
            pi = self.forward(state)              

        B, A, H, W = pi.shape
        flat_pi    = pi.permute(0, 2, 3, 1).reshape(-1, A)   
        dist       = torch.distributions.Categorical(flat_pi)
        action     = dist.sample()                             
        action_map = action.reshape(B, H, W)                  
        return action_map, pi

    def compute_gradient(self, state, action_map, advantage, entropy_coef=0.01):
        
        pi     = self.forward(state)              
        log_pi = torch.log(pi + 1e-8)

        B, A, H, W = pi.shape
        flat_lp  = log_pi.permute(0, 2, 3, 1).reshape(-1, A)   
        flat_idx = action_map.reshape(-1)                        

        selected = flat_lp[torch.arange(len(flat_idx), device=state.device), flat_idx]
        log_pi_a = selected.reshape(B, H, W)                    

        adv = advantage.detach()

        adv_std = adv.std()
        if adv_std > 1e-6:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)

        pg_loss = -(log_pi_a * adv).mean()

        entropy = -(pi * log_pi).sum(dim=1).mean()

        loss = pg_loss - entropy_coef * entropy
        return loss, pi


def reward_objective(st, st1, Ig):
    
    diff_before = (Ig - st ).pow(2).sum(dim=1).sqrt()   
    diff_after  = (Ig - st1).pow(2).sum(dim=1).sqrt()   
    return diff_before - diff_after                       


def reward_aesthetic(aesthetic_net, st, st1):
    
    to_pil = transforms.ToPILImage()
    with torch.no_grad():
        s_t  = aesthetic_net.score(to_pil(st [0].clamp(0, 1).cpu()))
        s_t1 = aesthetic_net.score(to_pil(st1[0].clamp(0, 1).cpu()))
    return float(s_t1 - s_t)


def compute_reward(aesthetic_net, st, st1, Ig):
    
    r_as_scalar = reward_aesthetic(aesthetic_net, st, st1)
    r_obj       = reward_objective(st, st1, Ig)           

    B, C, H, W  = st.shape
    r_as        = torch.full((B, H, W), r_as_scalar,
                             dtype=st.dtype, device=st.device)

    r = W1 * r_as + WD * r_obj
    return r   


def run_denoise_episode(policy, value, aesthetic, st, Ig, device):
    st = st.to(device)
    Ig = Ig.to(device)

    states  = [st]
    rewards = []

    policy.eval()
    for _ in range(N_STEPS_D):
        s = states[-1]

        action_map, _ = policy.select_action(s)
        action_map    = action_map.to(device)

        st1 = apply_action_map(s, action_map).to(device)  

        r = compute_reward(aesthetic, s, st1, Ig)          

        rewards.append(r.detach())
        states.append(st1.detach())

    policy.train()

    with torch.no_grad():
        V_sN = value(states[-1])    

    return states[:-1], rewards, V_sN   

if __name__ == "__main__":
    from aesthetic_net  import AestheticNet
    from value_network  import compute_total_reward

    device    = torch.device("cpu")
    aesthetic = AestheticNet(device=device)
    policy    = DenoisePolicyNetwork()
    value     = DenoiseValueNetwork()

    noisy = torch.rand(1, 3, 64, 64)
    clean = torch.rand(1, 3, 64, 64)

    states, rewards, V_sN = run_denoise_episode(
        policy, value, aesthetic, noisy, clean, device)

    print(f"States  : {len(states)} × {states[0].shape}")
    print(f"Rewards : {len(rewards)} × {rewards[0].shape}")
    print(f"V_sN    : {V_sN.shape},  mean={V_sN.mean().item():.4f}")

    from value_network import compute_total_reward, compute_advantage

    Rt = compute_total_reward(rewards, V_sN, gamma=GAMMA_D)
    print(f"Rt      : {Rt.shape},  mean={Rt.mean().item():.4f}")

    for t in range(N_STEPS_D):
        s_t        = states[t]
        action_map, _ = policy.select_action(s_t)
        Vst        = value(s_t)
        G          = compute_advantage(Rt, Vst)
        pl, _      = policy.compute_gradient(s_t, action_map, G, entropy_coef=0.01)
        vl, _      = value.compute_gradient(s_t, Rt)
        print(f"  step {t}: pl={pl.item():.4f}  vl={vl.item():.6f}  G.mean={G.mean().item():.4f}")

    print(" Test passed.")