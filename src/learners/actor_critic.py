from os.path import join

import torch
from torch import optim

from components.agents import REGISTRY


class ActorCritic:
    def __init__(self, args, runner, logger):
        # 保存参数
        self.args = args
        self.runner = runner
        self.logger = logger

        self.critic = REGISTRY[args.critic](args, self.runner.controller)

        self.actor_optimizer = optim.Adam(self.runner.controller.agent.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def can_learn(self):
        return self.runner.episode_done and self.runner.buffer.can_sample(granularity="episode")

    def learn(self):
        raise NotImplementedError("ActorCritic.learn is not implemented yet.")

    def cuda(self):
        self.runner.controller.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.runner.controller.save_models(path)
        self.critic.save_models(path, model_name="critic.th")
        torch.save(self.actor_optimizer.state_dict(), join(path, "actor_optimizer.th"))
        torch.save(self.critic_optimizer.state_dict(), join(path, "critic_optimizer.th"))

    def load_models(self, path):
        self.runner.controller.load_models(path)
        self.critic.load_models(path, model_name="critic.th")

        actor_optimizer_state = torch.load(join(path, "actor_optimizer.th"), map_location=torch.device("cpu"))
        critic_optimizer_state = torch.load(join(path, "critic_optimizer.th"), map_location=torch.device("cpu"))
        self.actor_optimizer.load_state_dict(actor_optimizer_state)
        self.critic_optimizer.load_state_dict(critic_optimizer_state)
