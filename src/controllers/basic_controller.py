import torch

from .abstract_controller import AbstractController
from components.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_selector_REGISTRY


# the controllers for vanilla value-based and policy-base rl algorithm
class BasicController(AbstractController):
    def __init__(self, args, env_scheme):
        super().__init__(args, env_scheme)

        # 保存args参数和scheme
        self.args = args
        self.env_scheme = env_scheme

        ## 初始化动作选择器
        self.action_selector = action_selector_REGISTRY[args.action_selector](args)

        ## Todo 构建builder用于构建input输入，比如拼接历史动作，比如拼接过去的帧，不再使用build_input函数

        self.builder = None

        ## 构建agent,agent的构造既依赖于builder(输入)，也依赖于env_scheme
        self.agent = agent_REGISTRY[self.args.agent](self.args, env_scheme)

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
