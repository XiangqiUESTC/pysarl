import torch

from buffers import REGISTRY as buffer_REGISTRY
from controllers import REGISTRY as controller_REGISTRY
from envs import REGISTRY as env_REGISTRY


class EpisodeRunner:
    """
    一次只收集一个完整回合。
    """

    def __init__(self, args, logger):
        self.env = None
        self.controller = None
        self.buffer = None
        self.env_scheme = None
        self.buffer_scheme = None

        self.args = args
        self.logger = logger

        self.t_env = 0
        self.t = 0
        self.episode = 0
        self.batch_size = 1

        self.setup()

    def run(self, test_mode=False):
        self.reset()
        self.buffer.new_episode()

        terminated = self.env.terminated
        truncated = self.env.terminated
        total_reward = 0
        state = self.env.get_state()

        while not (terminated or truncated):
            self.buffer.insert(state, "state", self.t)

            current_input = self.controller.builder.build_timestep(self.buffer, self.t)
            self.buffer.insert(current_input, "input", self.t)

            current_data = self.buffer.get_episode_data()

            action = self.controller.select_action(current_data, self.t_env, self.t, test_mode=test_mode)

            self.buffer.insert(action[0], "action", self.t)
            self.buffer.insert(terminated, "terminated", self.t)
            self.buffer.insert(truncated, "truncated", self.t)
            self.buffer.insert(torch.tensor([1.0]), "filled", self.t)

            state, reward, terminated, truncated, *_ = self.env.step(action.item())
            total_reward += reward
            self.buffer.insert(reward, "reward", self.t)

            self.t += 1

        self.buffer.insert(state, "state", self.t)
        self.buffer.insert(terminated, "terminated", self.t)
        self.buffer.insert(truncated, "truncated", self.t)

        final_input = self.controller.builder.build_timestep(self.buffer, self.t)
        self.buffer.insert(final_input, "input", self.t)

        if test_mode:
            self.buffer.discard_last_episode()
            return total_reward

        self.episode += 1
        self.t_env += self.t

        self.logger.log_scalar("total_rewards", total_reward, self.t_env)

        if self.episode % self.args.log_interval == 0:
            self.logger.logger.info(
                f"Episode: {self.episode:>5} t_env: {self.t_env:>10} total_reward: {total_reward}",
            )

        return total_reward

    def setup(self):
        self.env = env_REGISTRY[self.args.env](self.args)
        self.env_scheme = self.env.get_scheme()
        self.controller = controller_REGISTRY[self.args.controller](self.args, self.env_scheme)
        self.buffer_scheme = self.controller.get_buffer_scheme()
        self.buffer = buffer_REGISTRY[self.args.buffer](self.args, self.buffer_scheme)

    def reset(self):
        self.env.reset()
        self.t = 0
