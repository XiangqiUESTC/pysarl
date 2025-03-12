import gymnasium as gym
import numpy as np
from .single_agent_env import SingleAgentEnv


class Gym(SingleAgentEnv):
    def get_max_episode_steps(self):
        return self.game.spec.max_episode_steps

    def __init__(self, env_args):
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

    def step(self, action):
        state, reward, terminated, info, *_ = self.game.step(action)
        self.state = state
        return state, reward, terminated, info

    def reset(self):
        self.state, *_ = self.game.reset()
        self.terminated = False

    def get_state(self):
        return self.state

    def get_state_shape(self):
        return self.get_space_shape(self.game.observation_space)

    def get_action_shape(self):
        return self.get_space_shape(self.game.action_space)

    # 用来获得动作或者状态空间的维度
    def get_space_shape(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return space.shape
        else:
            assert False, "The space of game {}'s type is not considered yet!".format(self.env_args.game_name)

    def get_shape_dim(self, shape):
        if isinstance(shape, tuple):
            return len(shape)
        elif isinstance(shape, int) or isinstance(shape, np.integer):
            return 1
        else:
            assert False, "The space's dim of game {}'s type is not considered yet!".format(self.env_args.game_name)

    def get_env_info(self):
        state_shape = self.get_state_shape()
        action_shape = self.get_action_shape()
        return {
            "state_shape": state_shape,
            "state_dim": self.get_shape_dim(state_shape),
            "action_shape": action_shape,
            "action_dim": self.get_shape_dim(action_shape),
        }

    def render(self):
        pass

    def close(self):
        pass

