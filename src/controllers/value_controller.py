import numpy as np
import torch

from .abstract_controller import AbstractController
from modules.critic import REGISTRY as CRITIC_REGISTRY
from components.action_selectors import REGISTRY as ACTION_SELECTOR_REGISTRY


# the controllers for value-based
class ValueController(AbstractController):
    def __init__(self, args, scheme):
        super().__init__(args, scheme)

        self.args = args

        self.state_shape = scheme['state_info']["shape"]
        self.state_dim = scheme['state_info']["dim"]
        self.action_shape = scheme['action_info']["shape"]

        self._build_agent()
        self.action_selector = ACTION_SELECTOR_REGISTRY[args.action_selector](args)

    def select_action(self, batch, t_env, t, test_mode=False):
        flat_states = self._build_inputs(batch)
        agents_outputs = self.forward(flat_states)
        return self.action_selector.select_action(agents_outputs, t_env, t, test_mode=test_mode)

    def forward(self, flat_states):
        return self.agent(flat_states)

    def _build_agent(self):
        state_size = np.prod(self.state_shape)
        action_size = np.prod(self.action_shape)
        self.agent = CRITIC_REGISTRY[self.args.critic](state_size, action_size)

    def _build_inputs(self, batch):
        batch = torch.tensor(batch, dtype=torch.float32)
        flat_states = torch.flatten(batch, start_dim=-self.state_dim)
        return flat_states
