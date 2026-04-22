import torch


class Sarsa:
    def __init__(self, args, runner, logger):
        self.args = args
        self.runner = runner
        self.logger = logger
        self.last_learn_location = None

    def can_learn(self):
        latest_location = self.runner.buffer.get_latest_sampleable_step_location()
        return latest_location is not None and latest_location != self.last_learn_location

    def learn(self):
        latest_location = self.runner.buffer.get_latest_sampleable_step_location()
        if latest_location is None:
            return

        batch = self.runner.buffer.sample_latest_step()
        table = self.runner.controller.agent.table

        actions = batch["action"][:, 0:1].long()
        next_actions = batch["action"][:, 1:2].long()
        rewards = batch["reward"][:, 0:1].to(table.dtype)
        terminated = batch["terminated"][:, 1:2].to(table.dtype)

        current_q = self.runner.controller.forward(batch, t=0).to(table.dtype)
        chosen_action_q = current_q.gather(2, actions)

        with torch.no_grad():
            next_q = self.runner.controller.forward(batch, t=1).to(table.dtype)
            next_action_q = next_q.gather(2, next_actions)
            td_target = rewards + self.args.gamma * next_action_q * (1 - terminated)
            td_error = td_target - chosen_action_q

            state_indices = batch["input"][:, 0].reshape(-1).long()
            action_indices = actions.reshape(-1)
            table[state_indices, action_indices] += self.args.lr * td_error.reshape(-1)

        self.last_learn_location = latest_location
        self.logger.log_stats("td_error_abs", td_error.abs().mean().item(), self.runner.t_env)

    def cuda(self):
        self.runner.controller.cuda()

    def save_models(self, path):
        self.runner.controller.save_models(path)

    def load_models(self, path):
        self.runner.controller.load_models(path)
