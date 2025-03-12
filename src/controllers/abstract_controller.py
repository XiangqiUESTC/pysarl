from abc import ABC
from abc import abstractmethod


# 用来控制单智能体行为的抽象类
class AbstractController(ABC):
    def __init__(self, scheme, args):
        self.scheme = scheme
        self.args = args
        self.agent = None

    """
        method to select action
    """
    @abstractmethod
    def select_action(self, ):
        pass

    """
        method to calculate agent output
    """
    @abstractmethod
    def forward(self,):
        pass

    """
        inner method for building agent
    """
    @abstractmethod
    def _build_agent(self, input_shape):
        pass

    @abstractmethod
    def _build_inputs(self, batch):
        pass
