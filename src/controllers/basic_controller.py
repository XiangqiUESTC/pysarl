import torch

from .abstract_controller import AbstractController
from components.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_selector_REGISTRY
from components.state_encoder import REGISTRY as state_encoder_REGISTRY


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

        ## 构建agent
        self.agent = agent_REGISTRY[self.args.agent](self.args, self.scheme)

        ## 初始化动作选择器
        self.action_selector = action_selector_REGISTRY[args.action_selector](args)

        ## 构建state_encoder
        encoder_reg_name = getattr(args, "encoder", args.env_args.game_name)
        if encoder_reg_name in state_encoder_REGISTRY:
            self.state_encoder = state_encoder_REGISTRY[encoder_reg_name](self.args, self.scheme)
        else:
            self.state_encoder = None

    def select_action(self, transaction_batch, t_env, t, test_mode=False):
        agent_inputs = self._build_inputs(transaction_batch, t)
        agents_outputs = self.forward(agent_inputs)
        return self.action_selector.select_action(agents_outputs, t_env, t, test_mode=test_mode)

    def forward(self, states):
        return self.agent(states)

    def parameters(self):
        return self.agent.parameters()

    def _build_inputs(self, transaction_batch, t):
        states = transaction_batch["states"]
        some_state = states[t:t + 1]

        return torch.stack(some_state)

    def cuda(self):
        self.agent.cuda()
