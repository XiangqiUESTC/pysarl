import gymnasium as gym
import numpy as np
import torch

from .single_agent_env import SingleAgentEnv
from .specs.scheme import Scheme


class Gym(SingleAgentEnv):
    def __init__(self, env_args):
        """
            初始化方法
        """
        self.env_args = env_args
        self.game_name = env_args.game_name
        if self.game_name.startswith("ALE/"):
            import ale_py
            gym.register_envs(ale_py)
            self.game = gym.make(self.game_name)
        else:
            self.game = gym.make(self.game_name)
        # state设置为None
        self.state = None
        # 状态是否结束设置为None
        self.terminated = None
        # 获取scheme
        self.scheme = self.get_scheme()
        # 重置环境
        self.reset()

    def get_scheme(self):
        """
            获取模式,返回一个Scheme类
        """
        observation_space = self.game.observation_space
        action_space = self.game.action_space
        # 先处理状态空间
        observation_space_info = self._get_space_info(observation_space)
        action_space_info = self._get_space_info(action_space)
        return Scheme(*action_space_info, *observation_space_info)

    def get_max_episode_steps(self):
        return self.game.spec.max_episode_steps

    def step(self, action):
        state, reward, terminated, info, *_ = self.game.step(action)
        self.set_state(state)
        reward = torch.tensor(reward, dtype=torch.float)
        terminated = torch.tensor(terminated, dtype=torch.float)
        return self.get_state(), reward, terminated, info

    def reset(self):
        # 重置游戏，将self.state设置为初始状态
        state, *_ = self.game.reset()
        # 设置state
        self.set_state(state)
        # 将游戏结束设置为False
        self.terminated = False

    # 统一一下表达，如果state没有维度，那么给它加一个维度
    def set_state(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if len(state.shape) == 0:
            state = np.expand_dims(state, axis=0)
        state = torch.tensor(state)
        self.state = state

    def get_state(self):
        return self.state

    # 私有方法，用来获得一个space的信息，space是gym定义的几种基本的space的类型
    def _get_space_info(self, space):
        if isinstance(space, gym.spaces.Discrete):
            # 如果是离散类型，返回discrete字符和离散的个数
            return "discrete", space.n, None
        elif isinstance(space, gym.spaces.Box):
            # 如果是连续类型，continuous字符和维度
            return "continuous", None, torch.tensor(space.shape),
        else:
            # 还没有处理其他类型space的代码，遇到其它类型的代码就抛异常
            raise NotImplementedError(
                f"The space type {type(space)} defined by gym in game {self.env_args.game_name} "
                f"is not considered yet!"
            )

    def render(self):
        pass

    def close(self):
        pass
