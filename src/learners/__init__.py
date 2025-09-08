from .dqn import DQN
from .reinforce import Reinforce
from .actor_critic import ActorCritic

REGISTER = {
    "dqn": DQN,
    "reinforce": Reinforce,
    "actor_critic": ActorCritic
}