import torch

from .abstract_controller import AbstractController
from components.critics import REGISTRY as CRITIC_REGISTRY
from components.action_selectors.action_selectors import REGISTRY as ACTION_SELECTOR_REGISTRY


# the controllers for vanilla value-based and policy-base rl algorithm
class BasicController(AbstractController):
    def __init__(self, args, scheme):
        super().__init__(args, scheme)

        # 保存args参数和scheme
        self.args = args
        self.scheme = scheme

        # 从scheme中获得一些必要的参数
        if scheme.state_space_type == 'continuous':
            self.state_dim = len(scheme.continuous_state_shape)
        elif scheme.state_space_type == 'discrete':
            self.state_dim = 1
        else:
            raise NotImplementedError("State space type not supported!")

        # 构建agent
        self._build_agent()
        # 初始化动作选择器
        self.action_selector = ACTION_SELECTOR_REGISTRY[args.action_selector](args)

    def select_action(self, batch, t_env, t, test_mode=False):
        flat_states = self._build_inputs(batch)
        agents_outputs = self.forward(flat_states)
        return self.action_selector.select_action(agents_outputs, t_env, t, test_mode=test_mode)

    def forward(self, states):
        return self.agent(states)

    def _build_agent(self):
        self.agent = CRITIC_REGISTRY[self.args.critic](self.args, self.scheme)

    def _build_inputs(self, batch):
        batch = torch.tensor(batch, dtype=torch.float)
        flat_states = torch.flatten(batch, start_dim=-self.state_dim)
        return flat_states
