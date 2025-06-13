from .dqn import DQN
from .reinforce import Reinforce

REGISTER = {
    "dqn": DQN,
    "reinforce": Reinforce,
}