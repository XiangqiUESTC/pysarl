import torch

from .abstract_controller import AbstractController
from components.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_selector_REGISTRY


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

        ## 初始化动作选择器
        self.action_selector = action_selector_REGISTRY[args.action_selector](args)

        # 获取encoder的输出维度


        # 保存agent的输入维度
        self.input_dim = self.args.input_dim = self.get_agent_input_dim()

        ## 构建agent
        self.agent = agent_REGISTRY[self.args.agent](self.args, self.scheme)

    def select_action(self, transaction_batch, t_env, t, test_mode=False):
        agent_inputs = self.build_inputs(transaction_batch, t)
        agents_outputs = self.forward(agent_inputs)
        return self.action_selector.select_action(agents_outputs, t_env, t, test_mode=test_mode)

    def forward(self, states):
        return self.agent(states)

    def parameters(self):
        return self.agent.parameters()

    def build_inputs(self, transaction_batch, t):
        # 获取所有的state
        states = transaction_batch["states"]

        # 获取要多少历史state
        history_state_num = self.args.cat_history_state

        # 分两种情况，一种是可以直接拼接，一种是历史state不足，那么就要拼接空矩阵
        if t - history_state_num >= 0:
            some_state = states[:,t:t + 1]
        else:
            some_state = states[:,0:t + 1]

        return some_state

    def cuda(self):
        self.agent.cuda()

    def get_agent_input_dim(self):
        # Total state
        state_num = self.args.cat_history_state + 1
        return self.state_dim
