from abc import ABC, abstractmethod


class SingleAgentEnv(ABC):
    @abstractmethod
    def step(self, actions):
        """ 返回环境状态、奖励、是否结束等信息 """
        pass

    @abstractmethod
    def reset(self):
        """ 重置环境 """
        pass

    @abstractmethod
    def get_state(self):
        """ 获取当前状态 """
        pass

    @abstractmethod
    def get_state_shape(self):
        """ 获取状态的形状 """
        pass

    @abstractmethod
    def get_action_shape(self):
        """ 获取动作的形状 """

    @abstractmethod
    def render(self):
        """  """
        pass

    @abstractmethod
    def close(self):
        """  """
        pass

    @abstractmethod
    def get_env_info(self):
        pass

    @abstractmethod
    def get_max_episode_steps(self):
        pass
