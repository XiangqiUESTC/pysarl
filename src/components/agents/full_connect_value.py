import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn


class FullConnectedValue(nn.Module):
    """
    可配置的多层感知机价值网络。
    """

    def __init__(self, net_args, controller):
        super(FullConnectedValue, self).__init__()

        self.controller = controller
        self.env_scheme = controller.env_scheme
        self.input_space = controller.input_space
        self.net_args = net_args

        input_size, self.input_dim = self._get_input_shape(self.input_space)

        self.linear_layers = nn.ModuleList()

        n_layers = getattr(net_args, "n_layers", 2)
        hidden_dim = getattr(net_args, "hidden_dim")

        assert isinstance(n_layers, int) and n_layers > 0
        assert isinstance(hidden_dim, int) and hidden_dim > 0

        for layer_index in range(n_layers):
            if layer_index == 0:
                self.linear_layers.append(nn.Linear(input_size, hidden_dim))
            elif layer_index == n_layers - 1:
                self.linear_layers.append(nn.Linear(hidden_dim, 1))
            else:
                self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, batch, t=None):
        x = batch["input"] if t is None else batch["input"][:, t:t + 1]
        x = torch.flatten(x.float(), start_dim=-self.input_dim)

        for layer_index, layer in enumerate(self.linear_layers):
            x = layer(x)
            if layer_index < len(self.linear_layers) - 1:
                x = F.relu(x)

        return x

    def _get_input_shape(self, input_space):
        if isinstance(input_space, gym.spaces.Box):
            return int(torch.prod(torch.tensor(input_space.shape)).item()), len(input_space.shape)

        if isinstance(input_space, gym.spaces.Discrete):
            return 1, 1

        raise NotImplementedError(f"Input space type {type(input_space)} is not supported.")
