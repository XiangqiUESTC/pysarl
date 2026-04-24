import gymnasium as gym
import torch
from torch import nn


def get_module_device(module):
    return next(module.parameters()).device


def get_input_tensor(batch, t, device):
    states = batch["input"] if t is None else batch["input"][:, t:t + 1]
    return states.to(device=device, dtype=torch.float32)


class NatureCNNEncoder(nn.Module):
    """
    经典 Atari 卷积编码器，把 (C, H, W) 图像编码成一维特征。
    """

    def __init__(self, input_space):
        super(NatureCNNEncoder, self).__init__()

        if not isinstance(input_space, gym.spaces.Box) or len(input_space.shape) != 3:
            raise NotImplementedError("NatureCNNEncoder only supports image Box spaces with shape (C, H, W).")

        c, h, w = input_space.shape
        if h < 36 or w < 36:
            raise ValueError(
                f"NatureCNNEncoder expects reasonably sized images, got shape {input_space.shape}.",
            )

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            self.output_dim = self.conv(dummy).shape[1]

    def forward(self, states):
        leading_shape = states.shape[:-3]
        flat_states = states.reshape(-1, *states.shape[-3:])
        features = self.conv(flat_states)
        return features.reshape(*leading_shape, -1)
