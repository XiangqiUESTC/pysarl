import torch


class QTable:
    """
        Q表格类
    """
    def __init__(self, args, scheme):
        state_size = scheme.discrete_state_size
        action_size = scheme.discrete_action_size
        self.table = torch.full((state_size, action_size,), args.q_init_val, requires_grad=True, dtype=torch.float64)

    def __call__(self, *args, **kwargs):
        indexes = args[0].reshape(-1)
        return self.table[indexes.long()]

