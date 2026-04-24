import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import NatureCNNEncoder
from .nn_utils import get_input_tensor
from .nn_utils import get_module_device


class SimpleCNNCritic(nn.Module):
    """
    面向图像输入的离散动作价值网络。
    """

    def __init__(self, args, controller):
        super(SimpleCNNCritic, self).__init__()

        self.controller = controller
        self.env_scheme = controller.env_scheme
        self.input_space = controller.input_space

        action_space = self.env_scheme["action"]["space"]
        if not isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("SimpleCNNCritic only supports discrete action spaces.")

        if not isinstance(self.input_space, gym.spaces.Box) or len(self.input_space.shape) != 3:
            raise NotImplementedError("SimpleCNNCritic only supports image Box spaces with shape (C, H, W).")

        self.encoder = NatureCNNEncoder(self.input_space)
        self.fc1 = nn.Linear(self.encoder.output_dim, args.hidden_size)
        self.value_head = nn.Linear(args.hidden_size, action_space.n)

    def forward(self, batch, t=None):
        device = get_module_device(self)
        states = get_input_tensor(batch, t, device)
        features = self.encoder(states)
        x = F.relu(self.fc1(features))
        return self.value_head(x)

    def save_models(self, path):
        torch.save(self.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        state_dict = torch.load(f"{path}/agent.th", map_location=torch.device("cpu"))
        self.load_state_dict(state_dict)
