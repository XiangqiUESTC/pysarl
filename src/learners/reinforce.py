from os.path import join

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
        return_sample = None

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
            if return_sample is None:
                return_sample = self._build_discounted_returns(reward, valid)

            if self.args.baseline:
                state_values = self.critic(critic_batch).detach()
                loss = -(log_chosen_action_probs * (return_sample - state_values) * valid).sum() / valid.sum().clamp_min(1.0)
            else:
                loss = -(log_chosen_action_probs * return_sample * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            raise NotImplementedError("Method REINFORCE only has 3 kinds of formulas!")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.runner.buffer.clear()

    def cuda(self):
        self.runner.controller.cuda()

        if self.args.baseline:
            self.critic.cuda()

    def save_models(self, path):
        self.runner.controller.save_models(path)
        torch.save(self.optimizer.state_dict(), join(path, "agent_optimizer.th"))

        if self.args.baseline:
            self.critic.save_models(path, model_name="critic.th")
            torch.save(self.critic_optimizer.state_dict(), join(path, "critic_optimizer.th"))

    def load_models(self, path):
        self.runner.controller.load_models(path)
        agent_optimizer_state = torch.load(join(path, "agent_optimizer.th"), map_location=torch.device("cpu"))
        self.optimizer.load_state_dict(agent_optimizer_state)

        if self.args.baseline:
            self.critic.load_models(path, model_name="critic.th")
            critic_optimizer_state = torch.load(join(path, "critic_optimizer.th"), map_location=torch.device("cpu"))
            self.critic_optimizer.load_state_dict(critic_optimizer_state)

    def _build_discounted_returns(self, reward, valid):
        discounted_returns = torch.zeros_like(reward)
        running_return = torch.zeros_like(reward[:, 0])

        for t in range(reward.shape[1] - 1, -1, -1):
            running_return = (reward[:, t] + self.args.gamma * running_return) * valid[:, t]
            discounted_returns[:, t] = running_return

        return discounted_returns
