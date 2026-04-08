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
        pass

    def load_models(self, path):
        pass
