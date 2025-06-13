from copy import deepcopy
from torch import optim


class Reinforce:
    def __init__(self, args, scheme, controller, logger):
        self.args = args
        self.scheme = scheme
        self.controller = controller
        self.logger = logger


    def learn(self, buffer, t_env, episode_num):
        # 获取基本的数据结构
        # terminated和states的长度比actions和rewards和filled多一个

        # 需要所有的states
        states = buffer["states"]
        # terminated是用来判断下一状态是否为结束状态的，所以截取时不需要第一个
        terminated = buffer["terminated"][:,1:].float().unsqueeze(-1)

        # 最后一个动作、reward和filled都是无效的，填充的数据而已
        actions = buffer["actions"][:, :-1]
        rewards = buffer["rewards"][:, :-1].unsqueeze(-1)
        filled = buffer["filled"][:, :-1].unsqueeze(-1)

        pass

    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

