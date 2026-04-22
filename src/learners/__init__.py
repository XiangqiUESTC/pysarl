from .dqn import DQN
from .reinforce import Reinforce
from .actor_critic import ActorCritic
from .q_learning import QLearning
from .sarsa import Sarsa

REGISTER = {
    "dqn": DQN,
    "reinforce": Reinforce,
    "actor_critic": ActorCritic,
    "q_learning": QLearning,
    "sarsa": Sarsa,
}
