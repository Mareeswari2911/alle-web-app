import os
import time
import shutil
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from aesthetic_net  import AestheticNet
from policy_network import PolicyNetwork, apply_pac
from value_network  import ValueNetwork, compute_total_reward, compute_advantage

GAMMA         = 0.9
N_STEPS       = 6
LR            = 1e-4
MAX_EPOCHS    = 5000
BATCH_SIZE    = 4
IMAGE_SIZE    = 224
W0            = 1.0
W_FEA         = 0.2
W_EXP         = 0.2
E_TARGET      = 0.6
LAMBDA_SMOOTH = 0.2
PATCH_SIZE    = 16


CHECKPOINT_EVERY = 50          
PATIENCE         = 1000
MIN_DELTA        = 0.00001

_to_tensor = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])



class LOLDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return _to_tensor(img)



def _zip_checkpoints(save_dir, zip_path):
    """Zip the entire save_dir into zip_path (without extension)."""
    shutil.make_archive(zip_path, 'zip', save_dir)
    full = zip_path + '.zip'
    size_mb = os.path.getsize(full) / 1e6
    print(f"    Zipped checkpoints → {full}  ({size_mb:.1f} MB)")
    return full


def _upload_to_kaggle_dataset(zip_path, dataset_slug):

    try:
        import kaggle
        username, dataset_name = dataset_slug.split('/')
        # Create a temp dir with just the zip for upload
        tmp_dir = '/kaggle/working/_upload_tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        dest = os.path.join(tmp_dir, os.path.basename(zip_path))
        shutil.copy2(zip_path, dest)
        kaggle.api.dataset_create_version(
            tmp_dir,
            version_notes=f"Auto-save checkpoint",
            quiet=False,
            dir_mode='zip'
        )
        print(f"    Uploaded to Kaggle dataset: {dataset_slug}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        print(f"     Kaggle upload failed: {e}")
        print(f"      → Download manually from /kaggle/working/")


def _restore_from_zip(zip_path, save_dir):
    """Extract checkpoint zip into save_dir."""
    if os.path.exists(zip_path):
        print(f" Restoring checkpoints from {zip_path} ...")
        os.makedirs(save_dir, exist_ok=True)
        shutil.unpack_archive(zip_path, save_dir)
        print(f"    Restored: {os.listdir(save_dir)}")
        return True
    return False

def reward_aesthetic_subjective(aesthetic_net, st, st1):
    to_pil = transforms.ToPILImage()
    img_t  = st[0].detach().cpu().clamp(0, 1)
    img_t1 = st1[0].detach().cpu().clamp(0, 1)
    if img_t.dim() == 4:
        img_t = img_t.squeeze(0)
    if img_t1.dim() == 4:
        img_t1 = img_t1.squeeze(0)
    f_t  = aesthetic_net.score(to_pil(img_t))
    f_t1 = aesthetic_net.score(to_pil(img_t1))
    return torch.tensor(f_t1 - f_t, dtype=torch.float32)


def reward_feature_preservation(st1_batch, action_seq, lambda_smooth=LAMBDA_SMOOTH):
    B, C, H, W = st1_batch.shape
    R   = st1_batch[:, 0, :, :]
    G   = st1_batch[:, 1, :, :]
    Bch = st1_batch[:, 2, :, :]
    J_R = R.mean(dim=[1, 2])
    J_G = G.mean(dim=[1, 2])
    J_B = Bch.mean(dim=[1, 2])
    r_color  = ((J_R - J_G) ** 2 + (J_R - J_B) ** 2 + (J_G - J_B) ** 2).mean()
    r_smooth = 0.0
    for At in action_seq:
        grad_x   = (At[:, :, :, 1:] - At[:, :, :, :-1]).abs().mean()
        grad_y   = (At[:, :, 1:, :] - At[:, :, :-1, :]).abs().mean()
        r_smooth = r_smooth + grad_x + grad_y
    r_smooth = lambda_smooth * r_smooth / len(action_seq)
    return r_color + r_smooth


def reward_exposure_control(st1_batch, patch_size=PATCH_SIZE, E=E_TARGET):
    B, C, H, W = st1_batch.shape
    Y       = st1_batch.mean(dim=1, keepdim=True)
    patches = Y.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, -1, patch_size, patch_size)
    Y_b     = patches.mean(dim=[2, 3])
    return (Y_b - E).abs().mean()


def compute_immediate_reward_E(aesthetic_net, st, st1, action_seq,
                                w0=W0, w_fea=W_FEA, w_exp=W_EXP):
    r_as  = reward_aesthetic_subjective(aesthetic_net, st, st1)
    r_fea = reward_feature_preservation(st1, action_seq)
    r_exp = reward_exposure_control(st1)
    rt_E  = w0 * r_as - w_fea * r_fea - w_exp * r_exp
    B, C, H, W = st.shape
    return rt_E.expand(B, H, W)


def run_episode(policy_net, value_net, aesthetic_net, st, device):
    rewards     = []
    action_seqs = []
    states      = [st]

    for t in range(N_STEPS):
        s_current     = states[-1].to(device)
        action_idx, _ = policy_net.select_action(s_current)
        At_float      = (action_idx.float() / (policy_net.conv4.out_channels - 1))
        At_float      = At_float.unsqueeze(1).expand_as(s_current)
        st1           = apply_pac(s_current, action_idx)
        action_seqs.append(At_float.detach())
        rt = compute_immediate_reward_E(
            aesthetic_net, s_current, st1, [At_float]
        ).to(device)
        rewards.append(rt)
        states.append(st1.detach())

    s_N  = states[-1].to(device)
    V_sN = value_net(s_N)
    return states, rewards, action_seqs, V_sN


def _print_bar(epoch, done, total, p_loss, v_loss, bar_width=55):
    filled = int(bar_width * done / total)
    empty  = bar_width - filled
    bar    = '|' * filled + ' ' * empty
    line   = (f'Epoch {epoch:>5}: [{bar}] {done}/{total}'
              f'  p_loss={p_loss:.4f}  v_loss={v_loss:.4f}')
    if done == total:
        print('\r' + line, flush=True)
    else:
        print('\r' + line, end='', flush=True)


def train(
    dataset_path,
    save_dir='checkpoints',
    device=None,
    restore_zip=None,          
    kaggle_dataset_slug=None,  
    zip_every=50,              
):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        drive_dir = '/content/drive/MyDrive/LOL_checkpoints'
        os.makedirs(drive_dir, exist_ok=True)
        print(f" Google Drive mounted → {drive_dir}")
    except Exception:
        drive_dir = save_dir
        print(f"  Google Drive not available, saving locally → {drive_dir}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(drive_dir, exist_ok=True)


    _zip_candidates = [
        restore_zip,
        '/kaggle/input/alle-checkpoints1/checkpoints1_backup.zip',
        '/kaggle/input/checkpoints1/checkpoints1_backup.zip',
    ]
    for zc in _zip_candidates:
        if zc and os.path.exists(zc):
            _restore_from_zip(zc, drive_dir)
            break

    dataset         = LOLDataset(dataset_path)
    dataloader      = DataLoader(dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)
    n_batches_total = len(dataloader)
    print(f" Dataset: {len(dataset)} images | "
          f"Batches/epoch: {n_batches_total} | "
          f"Batch size: {BATCH_SIZE}")

    aesthetic_net = AestheticNet(device=device)
    policy_net    = PolicyNetwork().to(device)
    value_net     = ValueNetwork().to(device)
    opt_p         = optim.Adam(policy_net.parameters(), lr=LR)
    opt_v         = optim.Adam(value_net.parameters(),  lr=LR)

    best_v_loss      = float('inf')
    patience_counter = 0
    start_epoch      = 1

    latest_ckpt = os.path.join(drive_dir, 'latest_checkpoint.pt')
    if os.path.exists(latest_ckpt):
        print(f" Loading checkpoint from {latest_ckpt} ...")
        try:
            ckpt = torch.load(latest_ckpt, map_location=device)
            policy_net.load_state_dict(ckpt['policy_net'])
            value_net.load_state_dict(ckpt['value_net'])
            opt_p.load_state_dict(ckpt['opt_p'])
            opt_v.load_state_dict(ckpt['opt_v'])
            start_epoch      = ckpt['epoch'] + 1
            best_v_loss      = ckpt.get('best_v_loss', float('inf'))
            patience_counter = ckpt.get('patience_counter', 0)
            print(f" Resumed from epoch {start_epoch} | "
                  f"best_v_loss={best_v_loss:.4f} | "
                  f"patience={patience_counter}/{PATIENCE}")
        except Exception as e:
            print(f" Failed to load checkpoint: {e} — starting fresh")
    else:
        print(" Starting fresh training")

    print(f"\n{'='*60}")
    print(f"MAX_EPOCHS={MAX_EPOCHS} | PATIENCE={PATIENCE} | "
          f"BATCH={BATCH_SIZE} | LR={LR}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start  = time.time()
        epoch_p_loss = 0.0
        epoch_v_loss = 0.0
        n_batches    = 0

        for batch in dataloader:
            st = batch.to(device)

            states, rewards, action_seqs, V_sN = run_episode(
                policy_net, value_net, aesthetic_net, st, device
            )

            total_p_loss = torch.tensor(0.0, device=device)
            total_v_loss = torch.tensor(0.0, device=device)

            for t in range(N_STEPS):
                s_t           = states[t].to(device)
                action_idx, _ = policy_net.select_action(s_t)
                Rt            = compute_total_reward(rewards[t:], V_sN, gamma=GAMMA)
                V_st          = value_net(s_t)
                G             = compute_advantage(Rt, V_st)

                p_loss, _ = policy_net.compute_gradient(s_t, action_idx, G)
                v_loss, _ = value_net.compute_gradient(s_t, Rt)

                total_p_loss = total_p_loss + p_loss
                total_v_loss = total_v_loss + v_loss

            opt_p.zero_grad()
            total_p_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            opt_p.step()

            opt_v.zero_grad()
            total_v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
            opt_v.step()

            epoch_p_loss += total_p_loss.item()
            epoch_v_loss += total_v_loss.item()
            n_batches    += 1

            _print_bar(epoch, n_batches, n_batches_total,
                       total_p_loss.item(), total_v_loss.item())

        avg_p      = epoch_p_loss / n_batches
        avg_v      = epoch_v_loss / n_batches
        epoch_time = time.time() - epoch_start
        remaining  = (MAX_EPOCHS - epoch) * epoch_time
        eta_hours  = remaining / 3600

        _print_bar(epoch, n_batches_total, n_batches_total, avg_p, avg_v)
        print(f"     Time={epoch_time:.1f}s | ETA={eta_hours:.2f}h")

        ckpt_data = {
            'epoch'           : epoch,
            'policy_net'      : policy_net.state_dict(),
            'value_net'       : value_net.state_dict(),
            'opt_p'           : opt_p.state_dict(),
            'opt_v'           : opt_v.state_dict(),
            'best_v_loss'     : best_v_loss,
            'patience_counter': patience_counter,
        }

        torch.save(ckpt_data, latest_ckpt)

        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(drive_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(ckpt_data, ckpt_path)
            print(f"    Checkpoint saved → checkpoint_epoch_{epoch}.pt")

        if epoch % zip_every == 0:
            zip_base = '/kaggle/working/checkpoints1_backup'
            _zip_checkpoints(drive_dir, zip_base)
            if kaggle_dataset_slug:
                _upload_to_kaggle_dataset(zip_base + '.zip', kaggle_dataset_slug)
            else:
                print(f"    To download: open Kaggle file panel → "
                      f"working/checkpoints1_backup.zip → ⋮ → Download")

        if avg_v < best_v_loss - MIN_DELTA:
            best_v_loss      = avg_v
            patience_counter = 0
            best_path        = os.path.join(drive_dir, 'best_model.pt')
            torch.save({
                'epoch'     : epoch,
                'policy_net': policy_net.state_dict(),
                'value_net' : value_net.state_dict(),
            }, best_path)
            print(f"    Best model saved! v_loss={best_v_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    No improvement: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n Early stopping triggered at epoch {epoch}")
            print(f" Best v_loss: {best_v_loss:.4f}")
            # Final zip before exit
            _zip_checkpoints(drive_dir, '/kaggle/working/checkpoints1_backup')
            print("    Final zip created for download!")
            break

    print("\n Training Complete!")


if __name__ == "__main__":
    import numpy as np

    device = torch.device('cpu')

    aesthetic_net = AestheticNet(device=device)
    policy_net    = PolicyNetwork().to(device)
    value_net     = ValueNetwork().to(device)

    st = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    print("Running 1 episode (N=6 steps)...")
    states, rewards, action_seqs, V_sN = run_episode(
        policy_net, value_net, aesthetic_net, st, device
    )
    print(f"  states collected : {len(states)}")
    print(f"  rewards collected: {len(rewards)}")
    print(f"  reward[0] shape  : {rewards[0].shape}")
    print(f"  reward[0] value  : {rewards[0].mean().item():.4f}")

    Rt   = compute_total_reward(rewards, V_sN, gamma=GAMMA)
    V_st = value_net(st)
    G    = compute_advantage(Rt, V_st)
    print(f"  Rt shape    : {Rt.shape}")
    print(f"  G shape     : {G.shape}")

    r_as  = reward_aesthetic_subjective(aesthetic_net, st, states[1])
    r_fea = reward_feature_preservation(states[1], action_seqs[:1])
    r_exp = reward_exposure_control(states[1])
    print(f"  r_as  (Eq.10) : {r_as.item():.4f}")
    print(f"  r_fea (Eq.11) : {r_fea.item():.4f}")
    print(f"  r_exp (Eq.12) : {r_exp.item():.4f}")
    rt_E = W0 * r_as - W_FEA * r_fea - W_EXP * r_exp
    print(f"  rt_E  (Eq.13) : {rt_E.item():.4f}")

    action_idx, _ = policy_net.select_action(st)
    p_loss, _     = policy_net.compute_gradient(st, action_idx, G)
    v_loss, _     = value_net.compute_gradient(st, Rt)
    print(f"  policy loss (Eq.6): {p_loss.item():.4f}")
    print(f"  value  loss (Eq.4): {v_loss.item():.4f}")

    print("Test passed.")


