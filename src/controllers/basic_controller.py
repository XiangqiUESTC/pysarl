from copy import deepcopy

from builders import REGISTRY as builder_REGISTRY
from components.action_selectors import REGISTRY as action_selector_REGISTRY
from components.agents import REGISTRY as agent_REGISTRY
from utils.functions import get_space_shape_and_type


class BasicController:
    """
    统一协调构造器、智能体和动作选择器。
    """

    def __init__(self, args, env_scheme):
        self.args = args
        self.env_scheme = env_scheme

        ## 初始化动作选择器
        self.action_selector = action_selector_REGISTRY[args.action_selector](args)
        self.builder = builder_REGISTRY[self.args.builder](self.args, self.env_scheme)
        self.input_space = self.builder.output_space
        self.agent = agent_REGISTRY[self.args.agent](self.args, self)

    def get_buffer_scheme(self):
        buffer_scheme = deepcopy(self.env_scheme)
        input_shape, input_dtype = get_space_shape_and_type(self.input_space)

        buffer_scheme["input"] = {
            "shape": input_shape,
            "dtype": input_dtype,
            "space": self.input_space,
        }
        return buffer_scheme

    def select_action(self, batch_dict, t_env, t, test_mode=False):
        agent_outputs = self.forward(batch_dict, t=t)
        return self.action_selector.select_action(agent_outputs, t_env, t, test_mode=test_mode)

    def forward(self, batch_dict, t=None):
        if not isinstance(batch_dict, dict):
            raise TypeError(f"BasicController.forward expects a batch dict, got {type(batch_dict)}.")

        return self.agent(batch_dict, t=t)

    def parameters(self):
        return self.agent.parameters()

    def cuda(self):
        self.agent.cuda()
