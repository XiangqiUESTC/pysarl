

#
class BasicBuffer:
    def __init__(self, args, scheme):
        self.data = {}
        self.args = args
        self.scheme = scheme
        self.max_episode_length = 0

    def reset(self):
        pass

    def insert(self, transaction):
        if not isinstance(transaction, dict):
            raise TypeError("状态转移字典应为字典类型数据变量")
        else:
            for key, value in transaction.items():

                if key not in self.data:
                    self.data[key] = []
                self.data[key].append(value)

    def can_sample(self):
        pass

    def sample(self):
        pass