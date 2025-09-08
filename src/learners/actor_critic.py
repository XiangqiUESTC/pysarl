from copy import deepcopy
import torch
from torch import optim

from components.agents import REGISTRY

class ActorCritic:
    def __init__(self, args, scheme, controller, logger):
        # 保存参数
        self.args = args
        self.scheme = scheme
        self.controller = controller
        self.logger = logger

        # 定义critic
        self.critic = REGISTRY[args.critic](args, scheme)

        # 定义优化器
        self.actor_optimizer = optim.Adam(self.controller.agent.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)


    def learn(self, buffer, t_env, episode_num):
        pass

    def cuda(self):
        self.controller.cuda()
        self.critic.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
