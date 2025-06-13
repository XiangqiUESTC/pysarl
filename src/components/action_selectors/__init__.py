from .epsilon_greedy import EpsilonGreedy
from .policy_distribution import PolicyDistribution

REGISTRY={
    "epsilon_greedy": EpsilonGreedy,
    "policy_distribution": PolicyDistribution,
}