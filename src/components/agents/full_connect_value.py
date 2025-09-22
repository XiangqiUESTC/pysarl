"""
    全连接的V函数
"""
import torch
from torch import nn
import torch.nn.functional as F


class FullConnectedValue(nn.Module):
    def __init__(self, net_args, scheme):
        # 父类初始化
        super(FullConnectedValue, self).__init__()

        # 保存一些参数
        self.scheme = scheme
        self.net_args = net_args

        # 只用计算state的维度即可
        if scheme.state_space_type == 'continuous':
            state_size = torch.prod(scheme.continuous_state_shape)
            self.state_dim = len(scheme.continuous_state_shape)
        elif scheme.state_space_type == 'discrete':
            self.state_dim = 1
            state_size = 1
        else:
            raise NotImplementedError('Only discrete and discrete state space are supported!')
        # 线性层数量
        self.linear_lays = nn.ModuleList()

        n_layers = getattr(net_args, 'n_layers', 2)
        hidden_dim = getattr(net_args, 'hidden_dim')
        # 确保类型正确
        assert isinstance(n_layers, int) and n_layers > 0 and isinstance(hidden_dim, int) and hidden_dim > 0

        for i in range(n_layers):
            if i == 0:
                # 第一层
                self.linear_lays.append(nn.Linear(state_size, hidden_dim))
            elif i == n_layers-1:
                # 最后一层
                self.linear_lays.append(nn.Linear(hidden_dim, 1))
            else:
                self.linear_lays.append(nn.Linear(hidden_dim, hidden_dim))


    def forward(self, x):
        for i, layer in enumerate(self.linear_lays):
            x = layer(x)
            # 最后一层不使用激活函数
            if i < len(self.linear_lays) - 1:
                x = F.relu(x)
        return x