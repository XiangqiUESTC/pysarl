from copy import deepcopy

from torch import optim


class DQN:
    def __init__(self, args, runner, logger):
        self.args = args
        self.runner = runner
        self.logger = logger

        self.last_learn_t = 0
        self.learning_starts = getattr(args, "learning_starts", args.batch_size)
        self.learn_every_steps = getattr(args, "learn_every_steps", 1)
        self.target_update_interval = getattr(args, "target_update_interval", 1000)

        self.target_controller = deepcopy(self.runner.controller)
        self.last_target_update_t = 0

        self.param = list(self.runner.controller.parameters())
        self.optimizer = optim.Adam(self.param, lr=self.args.lr)

    def can_learn(self):
        has_enough_samples = self.runner.buffer.step_num >= self.learning_starts
        reaches_learn_interval = (self.runner.t_env - self.last_learn_t) >= self.learn_every_steps
        return has_enough_samples and reaches_learn_interval and self.runner.buffer.can_sample(granularity="step")

    def learn(self):
        batch = self.runner.buffer.sample(granularity="step")

        done = batch["terminated"][:, 1:].float()
        actions = batch["action"][:, :-1].long()
        rewards = batch["reward"][:, :-1]
        valid = batch["valid"][:, :-1]

        qs = self.runner.controller.forward(batch)
        online_q = qs[:, :-1]
        chosen_action_q_val = online_q.gather(2, actions)

        next_batch = {
            key: value[:, 1:]
            for key, value in batch.items()
        }

        if self.args.double_q:
            next_online_q = qs[:, 1:]
            next_actions = next_online_q.max(2)[1].unsqueeze(-1)
            next_target_q = self.target_controller.forward(next_batch).detach()
            max_next_q_value = next_target_q.gather(2, next_actions)
        else:
            target_q = self.target_controller.forward(next_batch).detach()
            max_next_q_value = target_q.max(2)[0].unsqueeze(-1)

        td_target = rewards + self.args.gamma * max_next_q_value * (1 - done)
        td_error = chosen_action_q_val - td_target
        masked_td_error = td_error * valid

        loss = (masked_td_error ** 2).sum() / valid.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_learn_t = self.runner.t_env

        self.logger.log_stats("loss", loss.item(), self.runner.t_env)

        if (self.runner.t_env - self.last_target_update_t) >= self.target_update_interval:
            self._update_target_network()
            self.last_target_update_t = self.runner.t_env

    def _update_target_network(self):
        self.target_controller.agent.load_state_dict(self.runner.controller.agent.state_dict())

    def cuda(self):
        self.runner.controller.cuda()
        self.target_controller.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
