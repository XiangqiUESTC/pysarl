import gymnasium as gym
import torch
from os.path import join


class QTable:
    """
    面向离散状态空间和离散动作空间的表格型 Q 函数。
    """

    def __init__(self, args, controller):
        input_space = controller.input_space
        action_space = controller.env_scheme["action"]["space"]

        if not isinstance(input_space, gym.spaces.Discrete):
            raise NotImplementedError("QTable only supports discrete builder output spaces.")

        if not isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("QTable only supports discrete action spaces.")

        state_size = input_space.n
        action_size = action_space.n

        self.table = torch.full(
            (state_size, action_size),
            args.q_init_val,
            requires_grad=True,
            dtype=torch.float64,
        )

    def __call__(self, batch, t=None):
        state_indexes = batch["input"] if t is None else batch["input"][:, t:t + 1]

        if state_indexes.shape[-1] == 1:
            state_indexes = state_indexes.squeeze(-1)

        leading_shape = state_indexes.shape
        flat_state_indexes = state_indexes.reshape(-1)
        q_values = self.table[flat_state_indexes.long()]
        return q_values.reshape(*leading_shape, -1)

    def parameters(self):
        return [self.table]

    def cuda(self):
        self.table = self.table.cuda()

    def save_models(self, path):
        torch.save(self.table.detach().cpu(), join(path, "agent.th"))

    def load_models(self, path):
        device = self.table.device
        loaded_table = torch.load(join(path, "agent.th"), map_location=torch.device("cpu"))
        self.table = loaded_table.to(device).requires_grad_(True)
