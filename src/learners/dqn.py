from copy import deepcopy

from os.path import join

import torch
import torch.nn.functional as F
from torch import optim

from utils.functions import move_batch_to_device


class DQN:
    def __init__(self, args, runner, logger):
        self.args = args
        self.runner = runner
        self.logger = logger

        self.last_learn_t = 0
        self.learn_count = 0
        self.learning_starts = getattr(args, "learning_starts", args.sample_step_num)
        self.learn_every_steps = getattr(args, "learn_every_steps", 1)
        self.target_update_interval = getattr(args, "target_update_interval", 1000)

        self.target_controller = deepcopy(self.runner.controller)

        self.param = list(self.runner.controller.parameters())
        self.optimizer = optim.Adam(self.param, lr=self.args.lr)

    def can_learn(self):
        has_enough_samples = self.runner.buffer.can_sample(
            granularity="step",
            sample_num=self.learning_starts,
        )
        reaches_learn_interval = (self.runner.t_env - self.last_learn_t) >= self.learn_every_steps
        return has_enough_samples and reaches_learn_interval and self.runner.buffer.can_sample(granularity="step")

    def learn(self):
        batch = self.runner.buffer.sample(granularity="step")
        batch = move_batch_to_device(batch, self.args.device)

        actions = batch["action"][:, 0:1].long()
        rewards = batch["reward"][:, 0:1]
        done = batch["terminated"][:, 1:2].float()

        online_q = self.runner.controller.forward(batch, t=0)
        chosen_action_q_val = online_q.gather(2, actions)

        with torch.no_grad():
            target_q = self.target_controller.forward(batch, t=1)
            max_next_q_value = target_q.max(dim=2, keepdim=True)[0]

        td_target = rewards + self.args.gamma * max_next_q_value * (1 - done)
        loss = F.mse_loss(chosen_action_q_val, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_learn_t = self.runner.t_env
        self.learn_count += 1

        self.logger.log_stats("loss", loss.item(), self.runner.t_env)

        if self.learn_count % self.target_update_interval == 0:
            self._update_target_network()

    def _update_target_network(self):
        self.target_controller.agent.load_state_dict(self.runner.controller.agent.state_dict())

    def cuda(self):
        self.runner.controller.cuda()
        self.target_controller.cuda()

    def save_models(self, path):
        self.runner.controller.save_models(path)
        torch.save(self.target_controller.agent.state_dict(), join(path, "target_agent.th"))
        torch.save(self.optimizer.state_dict(), join(path, "optimizer.th"))

    def load_models(self, path):
        self.runner.controller.load_models(path)
        target_state_dict = torch.load(join(path, "target_agent.th"), map_location=torch.device("cpu"))
        self.target_controller.agent.load_state_dict(target_state_dict)

        optimizer_state_dict = torch.load(join(path, "optimizer.th"), map_location=torch.device("cpu"))
        self.optimizer.load_state_dict(optimizer_state_dict)
