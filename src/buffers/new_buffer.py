import random
from collections import deque

import torch


class BasicBuffer:
    """
    用预分配好的整段回合张量来存储数据。
    """
    def __init__(self, args, data_scheme):
        self.args = args
        self.data_scheme = data_scheme

        self.max_episode_steps = args.env_args.max_episode_steps
        self.max_buffer_size = getattr(args, "buffer_size", None)
        self.default_sample_num = getattr(args, "batch_size", 1)

        self.episode_num = 0
        self.data = {}

        for key in self.data_scheme.keys():
            self.data[key] = deque(maxlen=self.max_buffer_size)

    def new_episode(self):
        for key, scheme in self.data_scheme.items():
            episode_shape = (1, self.max_episode_steps + 1) + scheme["shape"]
            new_tensor = torch.zeros(episode_shape, dtype=scheme["dtype"], device=self.args.device)
            self.data[key].append(new_tensor)

        if self.max_buffer_size is not None:
            self.episode_num = min(self.episode_num + 1, self.max_buffer_size)
        else:
            self.episode_num += 1

    def insert(self, item, key, t, episode_id=-1):
        target_slice = self.data[key][episode_id][0][t]
        item_tensor = torch.as_tensor(item, dtype=target_slice.dtype, device=target_slice.device)
        self.data[key][episode_id][0][t].copy_(item_tensor.reshape_as(target_slice))

    def get_episode_data(self, key, episode_id=-1):
        return self.data[key][episode_id]

    def sample(self, granularity="episode", sample_num=None):
        if granularity == "episode":
            return self.sample_episode(sample_num=sample_num)

        if granularity == "step":
            return self.sample_step(sample_num=sample_num)

        raise ValueError(f"Unsupported sample granularity: {granularity}")

    def can_sample(self, granularity="episode", sample_num=None):
        sample_num = self._resolve_sample_num(sample_num)

        if self.episode_num == 0:
            return False

        if granularity == "episode":
            return self.episode_num >= sample_num

        if granularity == "step":
            return len(self._collect_step_candidates()) >= sample_num

        raise ValueError(f"Unsupported sample granularity: {granularity}")

    def sample_episode(self, sample_num=None):
        self._assert_non_empty()

        sample_num = self._resolve_sample_num(sample_num)
        episode_indices = self._sample_indices(self.episode_num, sample_num)
        batch = self._build_batch_from_episode_indices(episode_indices)
        batch["valid"] = batch["filled"].clone()
        return batch

    def sample_step(self, sample_num=None):
        self._assert_non_empty()

        sample_num = self._resolve_sample_num(sample_num)
        step_candidates = self._collect_step_candidates()

        if len(step_candidates) == 0:
            raise RuntimeError("No valid step can be sampled from the buffer.")

        sampled_step_candidates = self._sample_items(step_candidates, sample_num)
        episode_indices = [episode_id for episode_id, _ in sampled_step_candidates]
        batch = self._build_batch_from_episode_indices(episode_indices)
        batch["valid"] = torch.zeros_like(batch["filled"])

        for batch_id, (_, step_index) in enumerate(sampled_step_candidates):
            batch["valid"][batch_id, step_index] = 1.0

        return batch

    def clear(self):
        for key in self.data.keys():
            self.data[key].clear()

        self.episode_num = 0

    def discard_last_episode(self):
        if self.episode_num == 0:
            return

        for key in self.data.keys():
            if len(self.data[key]) > 0:
                self.data[key].pop()

        self.episode_num -= 1

    def __getitem__(self, item):
        return self.data[item]

    def _resolve_sample_num(self, sample_num):
        if sample_num is None:
            return self.default_sample_num

        return sample_num

    def _assert_non_empty(self):
        if self.episode_num == 0:
            raise RuntimeError("The buffer is empty and cannot be sampled.")

    def _build_batch_from_episode_indices(self, episode_indices):
        batch = {}

        for key, episode_queue in self.data.items():
            batch[key] = torch.cat([episode_queue[index] for index in episode_indices], dim=0)

        return batch

    def _collect_step_candidates(self):
        step_candidates = []

        for episode_id in range(self.episode_num):
            filled = self.data["filled"][episode_id][0, :-1, 0]
            valid_step_indices = torch.nonzero(filled > 0, as_tuple=False).squeeze(-1).tolist()

            for step_index in valid_step_indices:
                step_candidates.append((episode_id, step_index))

        return step_candidates

    def _sample_indices(self, population_size, sample_num):
        candidates = list(range(population_size))
        return self._sample_items(candidates, sample_num)

    def _sample_items(self, candidates, sample_num):
        if sample_num <= len(candidates):
            return random.sample(candidates, sample_num)

        return random.choices(candidates, k=sample_num)
