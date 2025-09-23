from .dqn import DQN
from .reinforce import Reinforce
from .actor_critic import ActorCritic
from .ppo import PPO

REGISTER = {"dqn": DQN, "reinforce": Reinforce, "actor_critic": ActorCritic, "ppo": PPO}
