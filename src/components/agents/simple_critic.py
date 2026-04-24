import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import get_input_tensor
from .nn_utils import get_module_device


class SimpleCritic(nn.Module):
    """
    面向离散控制的简单多层感知机动作价值网络。
    """

    def __init__(self, args, controller):
        super(SimpleCritic, self).__init__()

        self.controller = controller
        self.env_scheme = controller.env_scheme
        self.input_space = controller.input_space

        action_space = self.env_scheme["action"]["space"]
        if not isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("SimpleCritic only supports discrete action spaces.")

        action_size = action_space.n
        input_size, self.input_dim = self._get_input_shape(self.input_space)

        self.fc1 = nn.Linear(input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_size)

    def forward(self, batch, t=None):
        device = get_module_device(self)
        states = get_input_tensor(batch, t, device)
        flat_states = torch.flatten(states, start_dim=-self.input_dim)

        x = F.relu(self.fc1(flat_states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_input_shape(self, input_space):
        if isinstance(input_space, gym.spaces.Box):
            return int(torch.prod(torch.tensor(input_space.shape)).item()), len(input_space.shape)

        if isinstance(input_space, gym.spaces.Discrete):
            return 1, 1

        raise NotImplementedError(f"Input space type {type(input_space)} is not supported.")

    def save_models(self, path):
        torch.save(self.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        state_dict = torch.load(f"{path}/agent.th", map_location=torch.device("cpu"))
        self.load_state_dict(state_dict)
