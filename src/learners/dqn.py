from copy import deepcopy

from torch import optim


class DQN:
    def __init__(self, args, runner, logger):
        self.args = args
        self.runner = runner
        self.logger = logger

        # 准备好目标网络
        self.target_controller = deepcopy(self.runner.controller)
        # 记录上次更新目标网络的时间
        self.last_target_update_episode = 0

        # 需要优化的参数和优化器
        self.param = list(self.runner.controller.parameters())
        self.optimizer = optim.Adam(self.param, lr=self.args.lr)

    def learn(self, buffer, t_env, episode_num):
        batch = buffer.sample(granularity="step")

        terminated = batch["terminated"][:, 1:].float()
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

        td_target = rewards + max_next_q_value * (1 - terminated)
        td_error = chosen_action_q_val - td_target
        masked_td_error = td_error * valid

        loss = (masked_td_error ** 2).sum() / valid.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.log_stats("loss", loss.item(), t_env)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_target_network()
            self.last_target_update_episode = episode_num

    def _update_target_network(self):
        self.target_controller.agent.load_state_dict(self.runner.controller.agent.state_dict())

    def cuda(self):
        self.runner.controller.cuda()
        self.target_controller.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
