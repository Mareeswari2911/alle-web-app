import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):


    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1,  kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        V = self.conv4(x)
        return V.squeeze(1)   # (B, H, W)

    def compute_gradient(self, state, Rt):

        V_st = self.forward(state)
        loss = F.mse_loss(V_st, Rt.detach())
        return loss, V_st


def compute_total_reward(rewards, V_sN, gamma=0.9):

    Rt = (gamma ** len(rewards)) * V_sN.detach()
    for r in reversed(rewards):
        Rt = r + gamma * Rt
    return Rt


def compute_advantage(Rt, V_st):

    return Rt - V_st.detach()


if __name__ == "__main__":
    net = ValueNetwork()
    x   = torch.rand(1, 3, 224, 224)
    V   = net(x)
    print("V(st) shape       :", V.shape)
    print("V(st) sample value:", V.mean().item())

    rewards = [torch.rand(1, 224, 224) * 0.1 for _ in range(6)]
    V_sN    = net(torch.rand(1, 3, 224, 224))
    Rt      = compute_total_reward(rewards, V_sN, gamma=0.9)
    print("Rt shape          :", Rt.shape)
    print("Rt sample value   :", Rt.mean().item())

    G = compute_advantage(Rt, V)
    print("Advantage G shape :", G.shape)
    print("G sample value    :", G.mean().item())

    loss, V_st = net.compute_gradient(x, Rt)
    print("Value loss        :", loss.item())
    print(" Test passed.")