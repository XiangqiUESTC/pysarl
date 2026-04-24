import gymnasium as gym
import torch
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
    def __init__(self,args, render=False):
        """
            初始化方法
        """
        self.args = args
        self.game_name = args.game_name
        self.device = args.device
        self.dtype = dtype_dict[args.dtype]
        self.render_enabled = render
        self.render_mode = getattr(args, "render_mode", "human")
        make_kwargs = self._build_make_kwargs()

        if self.render_enabled and "render_mode" not in make_kwargs:
            make_kwargs["render_mode"] = self.render_mode

        self._try_register_ale_envs()

        # 注册游戏并添加wrapper
        game = gym.make(self.game_name, **make_kwargs)

        max_episode_steps = args.max_episode_steps
        game = TimeLimit(game, max_episode_steps=max_episode_steps)

        # 添加wrapper
        for wrapper_name in args.env_wrappers:
            wrapper = REGISTRY[wrapper_name]
            game = wrapper(game)

        # 添加wrapper
        for wrapper_name in args.alg_env_wrappers:
            wrapper = REGISTRY[wrapper_name]
            game = wrapper(game)

        # 添加wrapper
        for wrapper_name in args.default_wrappers:
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
        return self.args.max_episode_steps

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

    def render(self):
        return self.game.render()

    def close(self):
        self.game.close()

    def _try_register_ale_envs(self):
        try:
            import ale_py
        except ImportError:
            return

        gym.register_envs(ale_py)

    def _build_make_kwargs(self):
        raw_make_kwargs = getattr(self.args, "make_kwargs", None)

        if raw_make_kwargs is None:
            return {}

        if isinstance(raw_make_kwargs, dict):
            return dict(raw_make_kwargs)

        if hasattr(raw_make_kwargs, "__dict__"):
            return vars(raw_make_kwargs).copy()

        raise TypeError(f"Unsupported make_kwargs type: {type(raw_make_kwargs)}")
