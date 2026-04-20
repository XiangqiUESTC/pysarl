import gymnasium as gym
import numpy as np
import torch


class BasicBuilder:
    """
    根据原始轨迹字段构造并缓存每个时间步的输入。
    当前 `input` 只由状态派生，动作仍保留在 batch 字典里。
    """

    def __init__(self, args, env_scheme):
        self.args = args
        self.env_scheme = env_scheme

        self.state_space = env_scheme["state"]["space"]
        self.state_shape = tuple(env_scheme["state"]["shape"])
        self.action_space = env_scheme["action"]["space"]

        self.history_frame_num = getattr(args, "history_frame_num", 0)
        self.history_action_num = getattr(args, "history_action_num", 0)

        if self.history_frame_num < 0:
            raise ValueError("history_frame_num must be non-negative.")

        if self.history_action_num < 0:
            raise ValueError("history_action_num must be non-negative.")

        self.output_space = self._build_output_space()

    def build_timestep(self, data_source, t):
        timestep_input = self._build_batched_timestep(data_source, t)

        if timestep_input.shape[0] != 1:
            raise ValueError(
                "build_timestep expects a single live episode. "
                "Use build_batch for batched inputs.",
            )

        return timestep_input[0]

    def build_batch(self, data_source, start_t=0, end_t=None):
        if isinstance(data_source, dict) and "input" in data_source:
            return data_source["input"][:, start_t:end_t]

        state_sequence = self._get_sequence(data_source, "state")

        if end_t is None:
            end_t = state_sequence.shape[1]

        built_steps = [self._build_batched_timestep(data_source, t) for t in range(start_t, end_t)]

        if len(built_steps) == 0:
            template_t = min(max(start_t, 0), state_sequence.shape[1] - 1)
            template = self._build_batched_timestep(data_source, template_t)
            return template.new_empty((template.shape[0], 0, *template.shape[1:]))

        return torch.stack(built_steps, dim=1)

    def _get_sequence(self, data_source, key, episode_id=-1):
        if isinstance(data_source, dict):
            return data_source[key]

        if hasattr(data_source, "get_episode_data"):
            return data_source.get_episode_data_by_key(key, episode_id)

        return data_source[key][episode_id]

    def _build_output_space(self):
        if self.history_frame_num == 0:
            return self.state_space

        if self._preserve_image_structure():
            channel_num, height, width = self.state_space.shape
            stacked_channels = channel_num * (self.history_frame_num + 1)
            low = np.repeat(self.state_space.low, self.history_frame_num + 1, axis=0)
            high = np.repeat(self.state_space.high, self.history_frame_num + 1, axis=0)

            return gym.spaces.Box(
                low=low,
                high=high,
                shape=(stacked_channels, height, width),
                dtype=self.state_space.dtype,
            )

        input_dim = int(np.prod(self.state_shape)) * (self.history_frame_num + 1)

        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(input_dim,),
            dtype=np.float32,
        )

    def _build_batched_timestep(self, data_source, t):
        if isinstance(data_source, dict) and "input" in data_source:
            return data_source["input"][:, t]

        state_sequence = self._get_sequence(data_source, "state")

        if self.history_frame_num == 0:
            return state_sequence[:, t]

        frame_tensors = []
        for offset in range(self.history_frame_num, -1, -1):
            index = t - offset
            if 0 <= index < state_sequence.shape[1]:
                state_tensor = state_sequence[:, index]
            else:
                state_tensor = state_sequence.new_zeros((state_sequence.shape[0], *state_sequence.shape[2:]))

            frame_tensors.append(state_tensor)

        if self._preserve_image_structure():
            return torch.cat(frame_tensors, dim=1)

        return torch.cat([self._flatten_state(frame_tensor) for frame_tensor in frame_tensors], dim=-1)

    def _flatten_state(self, state_tensor):
        return state_tensor.reshape(state_tensor.shape[0], -1)

    def _preserve_image_structure(self):
        return isinstance(self.state_space, gym.spaces.Box) and len(self.state_space.shape) == 3
