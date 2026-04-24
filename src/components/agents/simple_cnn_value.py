import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import NatureCNNEncoder
from .nn_utils import get_input_tensor
from .nn_utils import get_module_device


class SimpleCNNValue(nn.Module):
    """
    面向图像输入的状态价值网络。
    """

    def __init__(self, net_args, controller):
        super(SimpleCNNValue, self).__init__()

        self.controller = controller
        self.input_space = controller.input_space

        if not isinstance(self.input_space, gym.spaces.Box) or len(self.input_space.shape) != 3:
            raise NotImplementedError("SimpleCNNValue only supports image Box spaces with shape (C, H, W).")

        hidden_dim = getattr(net_args, "hidden_dim", getattr(net_args, "hidden_size", 256))

        self.encoder = NatureCNNEncoder(self.input_space)
        self.fc1 = nn.Linear(self.encoder.output_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch, t=None):
        device = get_module_device(self)
        states = get_input_tensor(batch, t, device)
        features = self.encoder(states)
        x = F.relu(self.fc1(features))
        return self.value_head(x)

    def save_models(self, path, model_name="critic.th"):
        torch.save(self.state_dict(), f"{path}/{model_name}")

    def load_models(self, path, model_name="critic.th"):
        state_dict = torch.load(f"{path}/{model_name}", map_location=torch.device("cpu"))
        self.load_state_dict(state_dict)
