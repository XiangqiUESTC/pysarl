from .policy_controller import PolicyController
from .value_controller import ValueController
REGISTRY = {
    'policy': PolicyController,
    'value': ValueController
}