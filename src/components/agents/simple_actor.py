import torch.nn as nn
import torch.nn.functional as F
import torch

# simple_full_connect_actor
class SimpleActor(nn.Module):
    def __init__(self, args, scheme):
        action_size = scheme.discrete_action_size
        if scheme.state_space_type == 'continuous':
            state_size = torch.prod(scheme.continuous_state_shape)
        elif scheme.state_space_type == 'discrete':
            state_size = 1
        else:
            raise NotImplementedError('Only discrete and discrete state space are supported!')
        super(SimpleActor, self).__init__()
        self.fc1 = nn.Linear(state_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_size)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(x)

