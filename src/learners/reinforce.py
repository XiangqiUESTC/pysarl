import torch
from torch import optim

from components.agents import REGISTRY


class Reinforce:
    def __init__(self, args, runner, logger):
        self.args = args
        self.runner = runner
        self.logger = logger

        controller = runner.controller

        if args.baseline:
            self.critic = REGISTRY[args.critic.name](args.critic, controller)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.optimizer = optim.Adam(controller.parameters(), lr=args.lr)

    def can_learn(self):
        return self.runner.episode_done and self.runner.buffer.can_sample(granularity="episode")

    def learn(self):
        batch = self.runner.buffer.sample(granularity="episode")

        action = batch["action"][:, :-1]
        reward = batch["reward"][:, :-1]
        valid = batch["valid"][:, :-1]

        action_probs = self.runner.controller.forward(batch)[:, :-1]
        chosen_action_probs = torch.gather(action_probs, -1, action)
        log_chosen_action_probs = torch.log(chosen_action_probs.clamp_min(1e-8))

        if self.args.formula == 1:
            episode_reward = (reward * valid).sum(dim=1).squeeze(-1)
            log_action_prob_sum = (log_chosen_action_probs * valid).sum(dim=1).squeeze(-1)
            loss = -(episode_reward * log_action_prob_sum).sum() / valid.sum().clamp_min(1.0)
        elif self.args.formula == 2:
            log_action_prob_cumsum = (log_chosen_action_probs * valid).cumsum(dim=1)
            loss = -(reward * log_action_prob_cumsum * valid).sum() / valid.sum().clamp_min(1.0)
        elif self.args.formula == 3:
            return_sample = self._build_discounted_returns(reward, valid)

            if self.args.baseline:
                critic_batch = {
                    key: value[:, :-1]
                    for key, value in batch.items()
                }
                state_values = self.critic(critic_batch).detach().clone()
                loss = -(log_chosen_action_probs * (return_sample - state_values) * valid).sum() / valid.sum().clamp_min(1.0)
            else:
                loss = -(log_chosen_action_probs * return_sample * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            raise NotImplementedError("Method REINFORCE only has 3 kinds of formulas!")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.runner.buffer.clear()

        if self.args.baseline:
            return_sample = self._build_discounted_returns(reward, valid)
            critic_batch = {
                key: value[:, :-1]
                for key, value in batch.items()
            }
            value_train_iters = getattr(self.args, "value_train_iters", 1)

            for _ in range(value_train_iters):
                value_prediction = self.critic(critic_batch)
                value_loss = ((return_sample - value_prediction) ** 2 * valid).sum() / valid.sum().clamp_min(1.0)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

    def cuda(self):
        self.runner.controller.cuda()

        if self.args.baseline:
            self.critic.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

    def _build_discounted_returns(self, reward, valid):
        discounted_returns = torch.zeros_like(reward)
        running_return = torch.zeros_like(reward[:, 0])

        for t in range(reward.shape[1] - 1, -1, -1):
            running_return = (reward[:, t] + self.args.gamma * running_return) * valid[:, t]
            discounted_returns[:, t] = running_return

        return discounted_returns
