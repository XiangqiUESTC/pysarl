import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import NatureCNNEncoder
from .nn_utils import get_input_tensor
from .nn_utils import get_module_device


class SimpleCNNActor(nn.Module):
    """
    面向图像输入的离散动作策略网络。
    """

    def __init__(self, args, controller):
        super(SimpleCNNActor, self).__init__()

        self.controller = controller
        self.env_scheme = controller.env_scheme
        self.input_space = controller.input_space

        action_space = self.env_scheme["action"]["space"]
        if not isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("SimpleCNNActor only supports discrete action spaces.")

        if not isinstance(self.input_space, gym.spaces.Box) or len(self.input_space.shape) != 3:
            raise NotImplementedError("SimpleCNNActor only supports image Box spaces with shape (C, H, W).")

        action_size = action_space.n
        self.encoder = NatureCNNEncoder(self.input_space)
        fc_input_size = self.encoder.output_dim
        self.fc1 = nn.Linear(fc_input_size, args.hidden_size)
        self.policy_head = nn.Linear(args.hidden_size, action_size)

    def forward(self, batch, t=None):
        device = get_module_device(self)
        states = get_input_tensor(batch, t, device)
        features = self.encoder(states)
        x = F.relu(self.fc1(features))
        x = self.policy_head(x)
        return F.softmax(x, dim=-1)

    def save_models(self, path):
        torch.save(self.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        state_dict = torch.load(f"{path}/agent.th", map_location=torch.device("cpu"))
        self.load_state_dict(state_dict)
