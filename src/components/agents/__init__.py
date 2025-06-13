from .simple_critic import SimpleCritic
from .q_table import QTable
from .simple_actor import SimpleActor

REGISTRY = {
    "simple_critic": SimpleCritic,
    "simple_actor": SimpleActor,
    "q_table": QTable,
}