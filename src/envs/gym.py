import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import TimeLimit

from utils.functions import get_space_shape_and_type
from wrapper import REGISTRY

dtype_dict = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


class Gym(gym.Env):
    def __init__(self,args):
        """
            初始化方法
        """
        self.args = args
        self.env_args = args.env_args
        self.game_name = args.env_args.game_name
        self.device = args.device
        self.dtype = dtype_dict[args.dtype]

        # 注册游戏并添加wrapper
        if self.game_name.startswith("ALE/"):
            import ale_py
            gym.register_envs(ale_py)
            game = gym.make(self.game_name)
        else:
            game = gym.make(self.game_name)

        max_episode_steps = args.env_args.max_episode_steps
        game = TimeLimit(game, max_episode_steps=max_episode_steps)

        # 添加wrapper
        for wrapper_name in args.default_wrappers:
            wrapper = REGISTRY[wrapper_name]
            game = wrapper(game)

        # 添加wrapper
        for wrapper_name in args.env_wrappers:
            wrapper = REGISTRY[wrapper_name]
            game = wrapper(game)

        self.game = game
        # state设置为None
        self._state = None
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
        ob_shape, ob_type = get_space_shape_and_type(self.game.observation_space)
        action_shape, action_type = get_space_shape_and_type(self.game.action_space)

        scheme = {
            "state":{
                "shape":ob_shape,
                "dtype":ob_type,
                "space": self.game.observation_space,
            },
            "action": {
                "shape": action_shape,
                "dtype": action_type,
                "space": self.game.action_space,
            },
            "reward":{
                "shape": (1,),
                "dtype": torch.float32,
            },
            "terminated":{
                "shape": (1,),
                "dtype": torch.float32,
            },
            "truncated":{
                "shape": (1,),
                "dtype": torch.float32,
            },
            "filled": {
                "shape": (1,),
                "dtype": torch.float32,
            }
        }
        return scheme

    def get_max_episode_steps(self):
        return self.args.env_args.max_episode_steps

    def step(self, action):
        state, reward, terminated, truncated ,info = self.game.step(action)
        self.set_state(state)
        self.terminated = terminated
        return state, reward, terminated, truncated ,info

    def reset(self, seed=None, options=None):
        # 重置游戏，将self.state设置为初始状态
        state, info = self.game.reset(seed=seed, options=options)
        self.set_state(state)
        self.terminated = False
        return state, info

    # 保存state
    def set_state(self, state):
        self._state = state
    # 获取state
    def get_state(self):
        return self._state

    # 私有方法，用来获得一个space的信息，space是gym定义的几种基本的space的类型
    def _get_space_shape_and_type(self, space):
        if isinstance(space, gym.spaces.Discrete):
            # 如果是离散类型，返回discrete字符和离散的个数
            return (1,), torch.int64
        elif isinstance(space, gym.spaces.Box):
            # 如果是连续类型，continuous字符和维度
            if np.issubdtype(space.dtype, np.integer):
                return space.shape, torch.int64
            return space.shape, torch.float32
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
