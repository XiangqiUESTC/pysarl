import time

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


        self.last_log_t = -self.args.log_interval - 1
        self.last_log_time = time.perf_counter()

        self.t_env = 0
        self.t = 0
        self.episode = 0
        self.batch_size = 1
        self.episode_active = False
        self.episode_done = False
        self.current_test_mode = False
        self.current_total_reward = 0
        self.current_state = None
        self.current_terminated = False
        self.current_truncated = False

        self.setup()

    def run(self, test_mode=False):
        episode_return = None

        while episode_return is None:
            step_result = self.step(test_mode=test_mode)
            if step_result["episode_done"]:
                episode_return = step_result["total_reward"]

        return episode_return

    def step(self, test_mode=False):
        if self.episode_active and self.current_test_mode != test_mode:
            raise RuntimeError("不能在一个episode的中途切换训练模式/测试模式")

        if not self.episode_active:
            self._start_episode(test_mode=test_mode)

        self.episode_done = False

        self.buffer.insert(self.current_state, "state", self.t)

        current_input = self.controller.builder.build_timestep(self.buffer, self.t)
        self.buffer.insert(current_input, "input", self.t)

        current_data = self.buffer.get_episode_data()

        action = self.controller.select_action(current_data, self.t_env, self.t, test_mode=test_mode)

        self.buffer.insert(action[0], "action", self.t)
        self.buffer.insert(self.current_terminated, "terminated", self.t)
        self.buffer.insert(self.current_truncated, "truncated", self.t)
        self.buffer.insert(torch.tensor([1.0]), "filled", self.t)

        next_state, reward, terminated, truncated, *_ = self.env.step(action.item())
        self.current_total_reward += reward
        self.buffer.insert(reward, "reward", self.t)

        self.current_state = next_state
        self.current_terminated = terminated
        self.current_truncated = truncated
        self.t += 1

        if not test_mode:
            self.t_env += 1

        if terminated or truncated:
            total_reward = self._finish_episode(test_mode=test_mode)
            return {
                "episode_done": True,
                "total_reward": total_reward,
            }

        return {
            "episode_done": False,
            "total_reward": None,
        }

    def setup(self):
        self.env = env_REGISTRY[self.args.env](self.args)
        self.env_scheme = self.env.get_scheme()
        self.controller = controller_REGISTRY[self.args.controller](self.args, self.env_scheme)
        self.buffer_scheme = self.controller.get_buffer_scheme()
        self.buffer = buffer_REGISTRY[self.args.buffer](self.args, self.buffer_scheme)

    def reset(self):
        self.env.reset()
        self.t = 0

    def _start_episode(self, test_mode=False):
        # 开始一个episode
        self.reset()
        self.buffer.new_episode()

        self.current_test_mode = test_mode
        self.current_total_reward = 0
        self.current_state = self.env.get_state()
        self.current_terminated = self.env.terminated
        self.current_truncated = self.env.terminated
        self.episode_active = True
        self.episode_done = False

    def _finish_episode(self, test_mode=False):
        # 结束一个episode
        # 做收尾工作
        self.buffer.insert(self.current_state, "state", self.t)
        self.buffer.insert(self.current_terminated, "terminated", self.t)
        self.buffer.insert(self.current_truncated, "truncated", self.t)

        final_input = self.controller.builder.build_timestep(self.buffer, self.t)
        self.buffer.insert(final_input, "input", self.t)

        total_reward = self.current_total_reward

        if test_mode:
            self.buffer.discard_last_episode()
        else:
            self.episode += 1

            self.logger.log_scalar("total_rewards", total_reward, self.t_env)

            if (self.t_env - self.last_log_t) // self.args.log_interval >= 1:
                current_time = time.perf_counter()
                frame_delta = self.t_env if self.last_log_t < 0 else self.t_env - self.last_log_t
                time_delta = max(current_time - self.last_log_time, 1e-8)
                fps = frame_delta / time_delta
                self.logger.logger.info(
                    f"Episode: {self.episode:>5} t_env: {self.t_env:>10} total_reward: {total_reward} fps: {fps:>8.2f}",
                )
                self.last_log_t = self.t_env
                self.last_log_time = current_time

        self.episode_active = False
        self.episode_done = True
        return total_reward
