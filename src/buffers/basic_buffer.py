from collections import deque


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
        pass