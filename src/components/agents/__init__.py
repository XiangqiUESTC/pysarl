from .simple_critic import SimpleCritic
from .q_table import QTable
from .simple_actor import SimpleActor
from .simple_cnn_actor import SimpleCNNActor
from .full_connect_value import FullConnectedValue

REGISTRY = {
    "simple_critic": SimpleCritic,
    "simple_actor": SimpleActor,
    "q_table": QTable,
    "simple_cnn_actor": SimpleCNNActor,
    "full_connect_value": FullConnectedValue
}