import gymnasium as gym
import torch


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

    def __call__(self, batch):
        state_indexes = batch["input"]

        if state_indexes.shape[-1] == 1:
            state_indexes = state_indexes.squeeze(-1)

        leading_shape = state_indexes.shape
        flat_state_indexes = state_indexes.reshape(-1)
        q_values = self.table[flat_state_indexes.long()]
        return q_values.reshape(*leading_shape, -1)
