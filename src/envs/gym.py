import gymnasium as gym
import numpy as np
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
        self.state = None
        self.terminated = None
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
        self.state = state
        return state, reward, terminated, info

    def reset(self):
        self.state, *_ = self.game.reset()
        self.terminated = False

    def get_state(self):
        return self.state

    # 用来获得一个space的信息，space是gym定义的几种基本的space的类型
    def _get_space_info(self, space):
        if isinstance(space, gym.spaces.Discrete):
            # 如果是离散类型，返回discrete字符和离散的个数
            return "discrete", space.n, None
        elif isinstance(space, gym.spaces.Box):
            # 如果是连续类型，continuous字符和维度
            return "continuous", None, space.shape,
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
