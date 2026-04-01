import torch
from gymnasium import ObservationWrapper, RewardWrapper, Wrapper



class StateToTensor(ObservationWrapper):
    """
        把state变成tensor
    """
    def __init__(self, env):
        super(StateToTensor, self).__init__(env)

    def observation(self, observation):
        return torch.tensor(observation)

class RewardToTensor(RewardWrapper):
    """
        把reward变成tensor
    """
    def __init__(self, env):
        super(RewardToTensor, self).__init__(env)

    def reward(self, reward):
        return torch.tensor(reward)

class FlagToTensor(Wrapper):
    """
        把truncate和terminate变成tensor
    """
    def __init__(self, env):
        super(FlagToTensor, self).__init__(env)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        terminated = torch.tensor(terminated, dtype=torch.float32)
        truncated = torch.tensor(truncated, dtype=torch.float32)
        return state, reward, terminated, truncated, info

class AllToTensor(Wrapper):
    def __init__(self, env):
        super(AllToTensor, self).__init__(env)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        state = torch.tensor(state)
        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated, dtype=torch.float32)
        truncated = torch.tensor(truncated, dtype=torch.float32)
        return state, reward, terminated, truncated, info