from .simple_critic import SimpleCritic
from .q_table import QTable

REGISTRY = {
    "simple_critic": SimpleCritic,
    "q_table": QTable,
}