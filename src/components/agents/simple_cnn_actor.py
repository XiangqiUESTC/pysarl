import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNActor(nn.Module):
    def __init__(self, args, scheme):
        super(SimpleCNNActor, self).__init__()
        action_size = scheme.discrete_action_size
        if scheme.state_space_type == 'continuous':
            state_shape = scheme.continuous_state_shape
            # 检测图像格式并自动调整CNN结构
            if len(state_shape) == 3:  # 图像格式 (C, H, W)
                c, h, w = state_shape
                self.state_dim = 3
                self.use_cnn = True

                # 根据图像尺寸调整卷积核大小
                conv_layers = []

                # 第一层卷积 - 使用更小的核以适应小尺寸输入
                kernel_h1 = min(3, h)  # 高度方向使用更小的核
                kernel_w1 = min(3, w)  # 宽度方向使用更小的核
                conv_layers.append(nn.Conv2d(c, 32, kernel_size=(kernel_h1, kernel_w1), stride=1))
                conv_layers.append(nn.ReLU())

                # 计算第一层输出尺寸
                h_out = h - kernel_h1 + 1
                w_out = w - kernel_w1 + 1

                # 第二层卷积
                kernel_h2 = min(2, h_out)
                kernel_w2 = min(2, w_out)
                conv_layers.append(nn.Conv2d(32, 64, kernel_size=(kernel_h2, kernel_w2), stride=1))
                conv_layers.append(nn.ReLU())

                # 计算第二层输出尺寸
                h_out = h_out - kernel_h2 + 1
                w_out = w_out - kernel_w2 + 1

                # 第三层卷积
                kernel_h3 = min(1, h_out)
                kernel_w3 = min(1, w_out)
                conv_layers.append(nn.Conv2d(64, 64, kernel_size=(kernel_h3, kernel_w3), stride=1))
                conv_layers.append(nn.ReLU())
                conv_layers.append(nn.Flatten())

                self.cnn = nn.Sequential(*conv_layers)

                # 计算CNN输出尺寸
                with torch.no_grad():
                    dummy = torch.zeros(1, c, h, w)
                    cnn_out = self.cnn(dummy)
                    fc_input_size = cnn_out.shape[1]

            else:  # 非图像格式
                self.use_cnn = False
                self.state_dim = len(state_shape)
                fc_input_size = torch.prod(torch.tensor(state_shape)).item()

        elif scheme.state_space_type == 'discrete':
            self.use_cnn = False
            self.state_dim = 1
            fc_input_size = 1
        else:
            raise NotImplementedError('Only discrete and continuous state space are supported!')



        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_size)

    def forward(self, states):
        # 处理图像输入
        if hasattr(self, 'use_cnn') and self.use_cnn:
            # 保存原始形状用于恢复
            original_shape = states.shape

            # 合并批次和时间维度 - 使用reshape代替view
            states = states.reshape(-1, *states.shape[-3:])

            # 通过CNN
            states = self.cnn(states)

            # 恢复原始批次和时间维度 - 使用reshape代替view
            states = states.reshape(*original_shape[:-3], -1)

        # 处理非图像输入
        else:
            # 使用flatten代替view
            states = torch.flatten(states, start_dim=-self.state_dim)

        # 通过全连接层
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)