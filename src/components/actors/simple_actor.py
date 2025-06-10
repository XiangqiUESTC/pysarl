import torch.nn as nn
import torch.nn.functional as F


# simple_full_connect_actor
class SimpleActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(SimpleActor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(x)

