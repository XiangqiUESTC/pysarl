import numpy as np
import torch
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper, spaces


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


class StateToDiscreteIndex(ObservationWrapper):
    """
        把由多个离散分量组成的 observation 压成单个离散编号。

        典型适用空间：
        1. gym.spaces.MultiDiscrete
        2. gym.spaces.Tuple，且其中每一项都必须是 gym.spaces.Discrete

        编码采用混合进制：
        - 每一维离散变量的基数由对应空间的 n 决定
        - 最终把多维离散状态映射到 [0, prod(n_i) - 1] 的单个整数

        例如 Blackjack 的 observation_space:
        Tuple(Discrete(32), Discrete(11), Discrete(2))
        会被压成 Discrete(32 * 11 * 2) = Discrete(704)

        这个 wrapper 的主要用途是给表格型算法提供单个离散状态索引。
    """

    def __init__(self, env):
        super(StateToDiscreteIndex, self).__init__(env)
        self._class_counts, self._starts = self._get_component_info(env.observation_space)
        total_state_num = int(np.prod(self._class_counts, dtype=np.int64))
        self.observation_space = spaces.Discrete(total_state_num)

    def observation(self, observation):
        flat_observation = self._to_int_array(observation)

        if flat_observation.size != len(self._class_counts):
            raise ValueError(
                f"Observation length {flat_observation.size} does not match space length {len(self._class_counts)}.",
            )

        adjusted_observation = flat_observation - self._starts
        if np.any(adjusted_observation < 0) or np.any(adjusted_observation >= self._class_counts):
            raise ValueError(
                f"Observation {flat_observation.tolist()} is out of bounds for counts "
                f"{self._class_counts.tolist()} and starts {self._starts.tolist()}.",
            )

        state_index = np.ravel_multi_index(
            tuple(adjusted_observation.tolist()),
            tuple(self._class_counts.tolist()),
        )
        return int(state_index)

    def _to_int_array(self, observation):
        if isinstance(observation, torch.Tensor):
            return observation.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)

        return np.asarray(observation, dtype=np.int64).reshape(-1)

    def _get_component_info(self, observation_space):
        if isinstance(observation_space, spaces.Discrete):
            return (
                np.asarray([observation_space.n], dtype=np.int64),
                np.asarray([observation_space.start], dtype=np.int64),
            )

        if isinstance(observation_space, spaces.MultiDiscrete):
            class_counts = np.asarray(observation_space.nvec, dtype=np.int64).reshape(-1)
            starts = np.asarray(observation_space.start, dtype=np.int64).reshape(-1)
            return class_counts, starts

        if isinstance(observation_space, spaces.Tuple) and all(
            isinstance(subspace, spaces.Discrete) for subspace in observation_space.spaces
        ):
            class_counts = np.asarray([subspace.n for subspace in observation_space.spaces], dtype=np.int64)
            starts = np.asarray([subspace.start for subspace in observation_space.spaces], dtype=np.int64)
            return class_counts, starts

        raise NotImplementedError(
            f"StateToDiscreteIndex does not support observation space {observation_space}.",
        )


class PongPreprocess(ObservationWrapper):
    """
        把 Atari Pong 的原始 RGB 帧预处理成适合神经网络输入的单通道图像。

        处理流程沿用经典 Pong policy gradient / DQN 常见的轻量预处理思路：
        1. 裁掉记分板和底部边框，只保留游戏区域
        2. 以 2 倍步长下采样到 80x80
        3. 只取单个颜色通道
        4. 把背景颜色清零，只保留球拍和小球
        5. 输出形状统一为 (1, 80, 80) 的 float32 张量语义

        之所以输出成 C,H,W 而不是 H,W，是为了和 builder 的历史帧堆叠逻辑对齐，
        这样 history_frame_num > 0 时可以直接沿通道维拼接，保持图像结构不被打平。

        这个 wrapper 是针对 Pong 原始画面设计的，不适合直接拿去处理其他 Atari 游戏。
    """

    def __init__(self, env):
        super(PongPreprocess, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, 80, 80),
            dtype=np.float32,
        )

    def observation(self, observation):
        if isinstance(observation, torch.Tensor):
            frame = observation.detach().cpu().numpy()
        else:
            frame = np.asarray(observation)

        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"PongPreprocess expects an RGB frame with shape (H, W, 3), got {frame.shape}.",
            )

        frame = frame[35:195]
        frame = frame[::2, ::2, 0]
        frame[(frame == 144) | (frame == 109)] = 0
        frame[frame != 0] = 1
        frame = frame.astype(np.float32, copy=False)
        return frame[None, :, :]


class PongMinimalActions(ActionWrapper):
    """
        把 Pong 的原始动作空间压成 2 个始终有效的动作：
        `RIGHTFIRE` 和 `LEFTFIRE`。

        这样 agent 从一开始就只需要学习“向上/向下打球”，
        不会再被 NOOP、FIRE、普通方向动作这些冗余动作干扰。
    """

    def __init__(self, env):
        super(PongMinimalActions, self).__init__(env)

        action_meanings = env.unwrapped.get_action_meanings()
        if len(action_meanings) < 6:
            raise ValueError(f"PongMinimalActions expects Pong-like 6-action env, got {action_meanings}.")

        self._action_map = [4, 5]
        self.action_space = spaces.Discrete(len(self._action_map))

    def action(self, action):
        action_id = int(action)
        if action_id < 0 or action_id >= len(self._action_map):
            raise ValueError(f"Action {action_id} is out of range for PongMinimalActions.")
        return self._action_map[action_id]


class AtariFrameSkipMax(Wrapper):
    """
        Atari 常用的动作重复 + 末两帧最大池化。

        作用有两个：
        1. 连续重复同一个动作若干帧，降低决策频率
        2. 对最后两帧做逐像素最大值，缓解 Atari 闪烁问题
    """

    def __init__(self, env, frame_skip=4):
        super(AtariFrameSkipMax, self).__init__(env)
        self.frame_skip = frame_skip
        self._obs_buffer = []

    def reset(self, seed=None, options=None):
        self._obs_buffer.clear()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        self._obs_buffer.clear()

        for step_id in range(self.frame_skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if step_id >= self.frame_skip - 2:
                self._obs_buffer.append(np.asarray(state))

            if terminated or truncated:
                break

        if len(self._obs_buffer) == 0:
            max_state = np.asarray(state)
        elif len(self._obs_buffer) == 1:
            max_state = self._obs_buffer[0]
        else:
            max_state = np.maximum(self._obs_buffer[0], self._obs_buffer[1])

        return max_state, total_reward, terminated, truncated, info


class AtariNoopReset(Wrapper):
    """
        在 reset 后执行若干次 NOOP，打散 Atari 初始状态。
    """

    def __init__(self, env, noop_max=30):
        super(AtariNoopReset, self).__init__(env)
        self.noop_max = noop_max

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)

        action_meanings = self.env.unwrapped.get_action_meanings()
        if not action_meanings or action_meanings[0] != "NOOP":
            return state, info

        noop_num = int(self.env.unwrapped.np_random.integers(1, self.noop_max + 1))
        for _ in range(noop_num):
            state, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                state, info = self.env.reset(seed=seed, options=options)

        return state, info


class AtariFireReset(Wrapper):
    """
        对需要 FIRE 才能开始的 Atari 游戏，在 reset 后自动执行启动动作。
    """

    def __init__(self, env):
        super(AtariFireReset, self).__init__(env)

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)

        action_meanings = self.env.unwrapped.get_action_meanings()
        if len(action_meanings) < 2 or action_meanings[1] != "FIRE":
            return state, info

        for action in (1, 2):
            state, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                state, info = self.env.reset(seed=seed, options=options)

        return state, info

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
