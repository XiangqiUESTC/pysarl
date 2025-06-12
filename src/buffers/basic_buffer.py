import random
from collections import deque
from copy import deepcopy

import torch


#
class BasicBuffer:
    def __init__(self, args, scheme):
        self.args = args
        self.scheme = scheme

        self.data = None
        self.max_episode_length = None
        self.episode_num = None

        self.reset()

    def reset(self):
        self.data = {}
        self.max_episode_length = 0
        self.episode_num = 0

    def insert(self, transaction):
        if not isinstance(transaction, dict):
            raise TypeError("状态转移字典应为字典类型数据变量")
        else:
            for key, value in transaction.items():
                if key not in self.data:
                    self.data[key] = deque(maxlen=self.args.buffer_size)
                self.data[key].append(value)

        if self.episode_num < self.args.buffer_size:
            self.episode_num += 1

    def can_sample(self):
        return self.episode_num >= self.args.buffer_size

    def sample(self):
        indices = random.sample(range(self.episode_num), self.args.batch_size)

        sampled_data = {}

        for key, dq in self.data.items():
            # 高效地从deque中提取指定索引的元素
            sampled_values = [torch.stack(dq[i]) for i in indices]
            # 创建新的固定大小的deque
            sampled_data[key] = sampled_values

        # 根据最长的步长补全数据
        max_episode_length = max([sum(self.data["filled"][i]) for i in indices]).item()

        padded_batch = {}

        # pad数据
        for key, tensor_batch in sampled_data.items():
            padded_batch[key] = []
            for tensor_to_pad in tensor_batch:
                print(f"tensor_to_pad_shape: {tensor_to_pad.shape}")
                # 获得形状
                pad_shape = list(tensor_to_pad.shape)
                # 在第一个维度进行填充
                pad_shape[0] = max_episode_length + 1 - pad_shape[0]
                print(f"pad_shape: {pad_shape}")
                # 创建用于填充的tensor
                pad_tensor = torch.zeros(pad_shape, dtype=tensor_to_pad.dtype)
                # 使用cat函数完成填充
                padded = torch.cat([tensor_to_pad, pad_tensor], dim=0)
                # 把填充后的tensor放回
                padded_batch[key].append(padded)
            padded_batch[key] = torch.stack(padded_batch[key])

        return  padded_batch

    def __getitem__(self, item):
        return self.data[item]