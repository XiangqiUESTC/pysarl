from collections import deque

import torch


class BasicBuffer:
    def __init__(self, args, env_scheme):
        self.args = args
        self.env_scheme = env_scheme
        self.valid_step = 0
        self.episode_num = 0
        # 指定了时间截断长度max_episode_steps
        self.max_episode_steps = max_episode_steps = args.env_args.max_episode_steps
        self.data = {}
        # 指定了buffer里面最多存储多少个episode
        self.max_buffer_size = getattr(args, "buffer_size", None)
        self.data_scheme = {}
        # 定义数据存储格式
        for key, sche in self.env_scheme.items():
            shape = sche["shape"]
            dtype = sche["dtype"]
            # +1是因为要存储终结状态
            shape = (1, max_episode_steps+1) + shape
            self.data_scheme[key] = {
                "shape": shape,
                "dtype": dtype,
            }
        # 初始化data
        for key in self.data_scheme.keys():
            self.data[key] = deque(maxlen=self.max_buffer_size)

    def new_episode(self):
        # 新增一个episode
        for key, scheme in self.data_scheme.items():
            new_tensor = torch.zeros(scheme["shape"], dtype=scheme["dtype"])
            self.data[key].append(new_tensor)
        # 更新episode计数器
        if self.max_buffer_size is not None:
            self.episode_num += 1
        else:
            self.episode_num = min(self.episode_num+1, self.max_episode_steps)

    def insert(self, item, key, t, episode_id=-1):
        """
            向一个episode的某项数据的t时刻插入一项数据
            episode_id : 默认向最后一个episode插入
        """
        self.data[key][episode_id][0][t] = item


    def clear(self):
        pass

    def __getitem__(self, item):
        return self.data[item]
