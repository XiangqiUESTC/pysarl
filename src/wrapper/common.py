import numpy as np
import torch
from gymnasium import ObservationWrapper, RewardWrapper, Wrapper, spaces


class StateToTensor(ObservationWrapper):
    """
        把state变成tensor
    """
    def __init__(self, env):
        super(StateToTensor, self).__init__(env)

    def observation(self, observation):
        return torch.tensor(observation)


class StateToOneHot(ObservationWrapper):
    """
        把类别型 observation 编码成 one-hot 浮点向量。

        这是一个 ObservationWrapper，只负责改写 observation 以及与之对应的
        observation_space，不会改动 action、reward、terminated、truncated
        等其他接口语义。

        当前支持的 observation_space 类型有三类：
        1. gym.spaces.Discrete
           单个离散状态编号，例如 0, 1, 2, ... , n-1。
           会被编码为长度为 n 的 one-hot 向量。

        2. gym.spaces.MultiDiscrete
           多个离散变量拼成的状态，每一维各自有不同的取值范围。
           会对每一维分别做 one-hot，再把所有 one-hot 结果按顺序拼接。

        3. gym.spaces.Tuple
           但要求 Tuple 里的每一项都必须是 gym.spaces.Discrete。
           如果 Tuple 里混入了 Box、MultiBinary 或其他空间类型，这个 wrapper
           不支持，会直接抛出 NotImplementedError。

        这个 wrapper 的核心用途是：
        把“离散类别状态”变成“神经网络更容易处理的浮点特征向量”。
        例如 DQN、REINFORCE 这类基于 MLP 的算法，通常更适合吃 one-hot 后
        的浮点输入，而不是直接吃一个类别编号。

        wrap 前后的语义变化如下：
        1. wrap 前 observation 可能是：
           - 一个整数
           - 一个多维离散数组
           - 一个由多个离散整数构成的 tuple

        2. wrap 后 observation 会统一变成：
           - 一个一维 numpy.float32 向量
           - 向量中的每一段都是对应原始离散变量的 one-hot 编码

        3. wrap 前 observation_space 可能是：
           - Discrete
           - MultiDiscrete
           - Tuple(Discrete, ...)

        4. wrap 后 observation_space 会被改写成：
           - Box(low=0.0, high=1.0, shape=(output_dim,), dtype=float32)

        一个具体例子：
        - 如果原始 observation_space 是 Tuple(Discrete(32), Discrete(11), Discrete(2))
        - 原始 observation 是 (18, 5, 1)
        - 那么会被编码成长度 32 + 11 + 2 = 45 的 one-hot 拼接向量

        需要注意：
        1. 这个 wrapper 只适合“类别型观测”，不适合连续 Box 观测，也不适合图像观测。
        2. 它通常应当放在把 observation 转成 tensor 的 wrapper 之前使用。
           例如先做 StateToOneHot，再做 AllToTensor。
        3. 本实现会把输入 observation 先展平成一维整数数组，再逐项编码；
           因而要求 observation 的实际结构必须和 observation_space 描述一致。
    """
    def __init__(self, env):
        super(StateToOneHot, self).__init__(env)
        self._space = env.observation_space
        self._class_counts = self._get_class_counts(self._space)
        output_dim = sum(self._class_counts)
        # one-hot 编码后，所有 observation 都会被统一表示成一维 float32 向量。
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(output_dim,),
            dtype=np.float32,
        )

    def observation(self, observation):
        """
            把单个原始 observation 编码为 one-hot 后的一维向量。

            编码流程是：
            1. 先把 observation 展平成一维整数数组
            2. 对数组中的每一项，根据对应类别数单独做 one-hot
            3. 把每一项的 one-hot 向量首尾拼接成最终输出
        """
        flat_observation = self._to_int_array(observation)

        if flat_observation.size != len(self._class_counts):
            raise ValueError(
                f"Observation length {flat_observation.size} does not match space length {len(self._class_counts)}."
            )

        encoded_parts = []
        for value, class_count in zip(flat_observation, self._class_counts):
            encoded = np.zeros(class_count, dtype=np.float32)
            encoded[int(value)] = 1.0
            encoded_parts.append(encoded)

        return np.concatenate(encoded_parts, axis=0)

    def _to_int_array(self, observation):
        """
            把原始 observation 统一转换成一维 int64 数组。

            这里同时兼容 numpy / Python 原生结构 / torch.Tensor，
            这样它既能处理环境直接返回的 observation，也能处理前面某些
            wrapper 已经转成 tensor 的情况。
        """
        if isinstance(observation, torch.Tensor):
            return observation.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)

        return np.asarray(observation, dtype=np.int64).reshape(-1)

    def _get_class_counts(self, observation_space):
        """
            根据 observation_space 解析出每个离散分量的类别数。

            返回结果是一个列表：
            - Discrete(n) -> [n]
            - MultiDiscrete([n1, n2, ...]) -> [n1, n2, ...]
            - Tuple(Discrete(a), Discrete(b), ...) -> [a, b, ...]
        """
        if isinstance(observation_space, spaces.Discrete):
            return [observation_space.n]

        if isinstance(observation_space, spaces.MultiDiscrete):
            return observation_space.nvec.astype(np.int64).tolist()

        if isinstance(observation_space, spaces.Tuple) and all(
            isinstance(subspace, spaces.Discrete) for subspace in observation_space.spaces
        ):
            return [subspace.n for subspace in observation_space.spaces]

        raise NotImplementedError(f"StateToOneHot does not support observation space {observation_space}.")

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

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        state = torch.tensor(state)
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        state = torch.tensor(state)
        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated, dtype=torch.float32)
        truncated = torch.tensor(truncated, dtype=torch.float32)
        return state, reward, terminated, truncated, info
