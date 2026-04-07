import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNActor(nn.Module):
    """
    使用卷积网络处理图像输入，并对向量输入提供多层感知机回退路径。
    """

    def __init__(self, args, controller):
        super(SimpleCNNActor, self).__init__()

        self.controller = controller
        self.env_scheme = controller.env_scheme
        self.input_space = controller.input_space

        action_space = self.env_scheme["action"]["space"]
        if not isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("SimpleCNNActor only supports discrete action spaces.")

        action_size = action_space.n

        if isinstance(self.input_space, gym.spaces.Box):
            state_shape = self.input_space.shape
            if len(state_shape) == 3:
                c, h, w = state_shape
                self.input_dim = 3
                self.use_cnn = True

                conv_layers = []

                kernel_h1 = min(3, h)
                kernel_w1 = min(3, w)
                conv_layers.append(nn.Conv2d(c, 32, kernel_size=(kernel_h1, kernel_w1), stride=1))
                conv_layers.append(nn.ReLU())

                h_out = h - kernel_h1 + 1
                w_out = w - kernel_w1 + 1

                kernel_h2 = min(2, h_out)
                kernel_w2 = min(2, w_out)
                conv_layers.append(nn.Conv2d(32, 64, kernel_size=(kernel_h2, kernel_w2), stride=1))
                conv_layers.append(nn.ReLU())

                h_out = h_out - kernel_h2 + 1
                w_out = w_out - kernel_w2 + 1

                kernel_h3 = min(1, h_out)
                kernel_w3 = min(1, w_out)
                conv_layers.append(nn.Conv2d(64, 64, kernel_size=(kernel_h3, kernel_w3), stride=1))
                conv_layers.append(nn.ReLU())
                conv_layers.append(nn.Flatten())

                self.cnn = nn.Sequential(*conv_layers)

                with torch.no_grad():
                    dummy = torch.zeros(1, c, h, w)
                    cnn_out = self.cnn(dummy)
                    fc_input_size = cnn_out.shape[1]
            else:
                self.use_cnn = False
                self.input_dim = len(state_shape)
                fc_input_size = int(torch.prod(torch.tensor(state_shape)).item())
        elif isinstance(self.input_space, gym.spaces.Discrete):
            self.use_cnn = False
            self.input_dim = 1
            fc_input_size = 1
        else:
            raise NotImplementedError("Only discrete and Box input spaces are supported.")

        self.fc1 = nn.Linear(fc_input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_size)

    def forward(self, batch, t=None):
        states = batch["input"] if t is None else batch["input"][:, t:t + 1]

        if self.use_cnn:
            original_shape = states.shape
            states = states.float().reshape(-1, *states.shape[-3:])
            states = self.cnn(states)
            states = states.reshape(*original_shape[:-3], -1)
        else:
            states = torch.flatten(states.float(), start_dim=-self.input_dim)

        # 动作保留在 batch 字典中，后续如果需要，
        # 可以在卷积编码之后再做融合，而不需要改控制器接口。
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)
