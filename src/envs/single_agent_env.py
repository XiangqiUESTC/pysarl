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
    def get_max_episode_steps(self):
        pass

    @abstractmethod
    def get_scheme(self):
        pass

    @abstractmethod
    def render(self):
        """  """
        pass

    @abstractmethod
    def close(self):
        """  """
        pass
