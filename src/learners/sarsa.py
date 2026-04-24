import torch


class Sarsa:
    def __init__(self, args, runner, logger):
        self.args = args
        self.runner = runner
        self.logger = logger
        self.last_learn_location = None
        self.batch_update = getattr(args, "batch_update", False)

    def can_learn(self):
        if self.batch_update:
            return self.runner.episode_done and self.runner.buffer.can_sample(granularity="episode")

        latest_location = self.runner.buffer.get_latest_sampleable_step_location()
        return latest_location is not None and latest_location != self.last_learn_location

    def learn(self):
        if self.batch_update:
            batch = self.runner.buffer.sample(granularity="episode")
            mean_abs_td_error, update_num = self._learn_from_episode_batch(batch)
            self.logger.log_stats("td_error_abs", mean_abs_td_error, self.runner.t_env)
            self.logger.log_stats("batch_update_num", update_num, self.runner.t_env)
            return

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

    def _learn_from_episode_batch(self, batch):
        table = self.runner.controller.agent.table
        state_indices, action_indices, rewards, next_state_indices, next_action_indices, terminated = self._build_transition_candidates(batch)
        update_num = state_indices.numel()

        if update_num == 0:
            return 0.0, 0

        abs_td_errors = []
        update_order = torch.randperm(update_num).tolist()

        with torch.no_grad():
            for idx in update_order:
                state_index = state_indices[idx].item()
                action_index = action_indices[idx].item()
                reward = rewards[idx].item()
                next_state_index = next_state_indices[idx].item()
                next_action_index = next_action_indices[idx].item()
                is_terminated = terminated[idx].item()

                current_q = table[state_index, action_index].item()
                next_action_q = table[next_state_index, next_action_index].item()
                td_target = reward + self.args.gamma * next_action_q * (1 - is_terminated)
                td_error = td_target - current_q
                table[state_index, action_index] += self.args.lr * td_error
                abs_td_errors.append(abs(td_error))

        return sum(abs_td_errors) / update_num, update_num

    def _build_transition_candidates(self, batch):
        transition_mask = batch["filled"][:, :-1].reshape(-1) > 0.5
        flat_state_indices = batch["input"][:, :-1].reshape(-1).long()
        flat_action_indices = batch["action"][:, :-1].reshape(-1).long()
        flat_rewards = batch["reward"][:, :-1].reshape(-1).to(torch.float64)
        flat_next_state_indices = batch["input"][:, 1:].reshape(-1).long()
        flat_next_action_indices = batch["action"][:, 1:].reshape(-1).long()
        flat_terminated = batch["terminated"][:, 1:].reshape(-1).to(torch.float64)

        return (
            flat_state_indices[transition_mask],
            flat_action_indices[transition_mask],
            flat_rewards[transition_mask],
            flat_next_state_indices[transition_mask],
            flat_next_action_indices[transition_mask],
            flat_terminated[transition_mask],
        )
