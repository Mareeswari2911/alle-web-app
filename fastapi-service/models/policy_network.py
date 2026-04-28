import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_LOW  = -0.5
ACTION_HIGH =  1.0
ACTION_STEP =  1 / 18
NUM_ACTIONS = round((ACTION_HIGH - ACTION_LOW) / ACTION_STEP) + 1


class PolicyNetwork(nn.Module):

    def __init__(self, in_channels=3, num_actions=NUM_ACTIONS):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, num_actions, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        pi = F.softmax(x, dim=1)
        return pi

    def select_action(self, state):
        with torch.no_grad():
            pi = self.forward(state)
        action_idx = pi.argmax(dim=1)
        return action_idx, pi

    def log_prob(self, state, action_idx):
        pi = self.forward(state)
        log_pi = torch.log(pi + 1e-8)
        B, A, H, W = log_pi.shape
        log_pi_flat = log_pi.permute(0, 2, 3, 1).reshape(-1, A)
        idx_flat    = action_idx.reshape(-1)
        selected    = log_pi_flat[torch.arange(len(idx_flat)), idx_flat]
        return selected.reshape(B, H, W), pi

    def compute_gradient(self, state, action_idx, advantage):
        log_pi, pi = self.log_prob(state, action_idx)
        loss = -(log_pi * advantage.detach()).mean()
        return loss, pi


def idx_to_action_value(action_idx):
    return ACTION_LOW + action_idx.float() * ACTION_STEP


def apply_pac(st, action_idx):
    At = idx_to_action_value(action_idx).unsqueeze(1)
    st1 = st + At * st * (1.0 - st)
    st1 = st1.clamp(0.0, 1.0)
    return st1


if __name__ == "__main__":
    import numpy as np
    net = PolicyNetwork()
    x   = torch.rand(1, 3, 224, 224)
    pi  = net(x)
    print("pi shape     :", pi.shape)
    print("pi sum (per pixel, should be 1):", pi.sum(dim=1).mean().item())
    action_idx, _ = net.select_action(x)
    print("action_idx shape:", action_idx.shape)
    st1 = apply_pac(x, action_idx)
    print("enhanced st+1 shape:", st1.shape)
    print("st+1 range: [{:.4f}, {:.4f}]".format(st1.min().item(), st1.max().item()))
    advantage = torch.rand(1, 224, 224)
    loss, _ = net.compute_gradient(x, action_idx, advantage)
    print("policy loss (Eq. 6):", loss.item())
    print("NUM_ACTIONS:", NUM_ACTIONS)
    print("Test passed.")