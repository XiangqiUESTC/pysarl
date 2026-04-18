import time

import torch

from buffers import REGISTRY as buffer_REGISTRY
from controllers import REGISTRY as controller_REGISTRY
from envs import REGISTRY as env_REGISTRY


class SerialRunner:
    """
    串行地与单个环境交互并收集完整回合。
    """
    def __init__(self, args, logger):
        self.env = None
        self.test_env = None
        self.controller = None
        self.buffer = None
        self.test_buffer = None
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
        self.train_reset_count = 0
        self.test_reset_count = 0

        self.setup()

    def run(self, test_mode=False):
        episode_return = None

        while episode_return is None:
            step_result = self.step(test_mode=test_mode)
            if step_result["episode_done"]:
                episode_return = step_result["total_reward"]

        return episode_return

    def step(self, test_mode=False):
        current_env = self.test_env if test_mode else self.env
        episode_buffer = self.test_buffer if test_mode else self.buffer

        if self.episode_active and self.current_test_mode != test_mode:
            raise RuntimeError("不能在一个episode的中途切换训练模式/测试模式")

        if not self.episode_active:
            self._start_episode(test_mode=test_mode)

        self.episode_done = False

        step_transaction = {"state": self.current_state}

        current_data = episode_buffer.get_episode_data()
        self._write_current_step(current_data, "state", self.current_state)

        current_input = self.controller.builder.build_timestep(episode_buffer, self.t)

        step_transaction["input"] = current_input

        self._write_current_step(current_data, "input", current_input)

        action = self.controller.select_action(current_data, self.t_env, self.t, test_mode=test_mode)

        step_transaction["action"] = action[0]

        step_transaction["terminated"] = self.current_terminated
        step_transaction["truncated"] = self.current_truncated
        step_transaction["filled"] = torch.tensor([1.0])

        next_state, reward, terminated, truncated, *_ = current_env.step(action.item())
        reward_value = float(reward)
        self.current_total_reward += reward_value

        step_transaction["reward"] = reward_value
        episode_buffer.insert_step_transaction(step_transaction, self.t)

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
        self.test_env = env_REGISTRY[self.args.env](self.args)
        self.env_scheme = self.env.get_scheme()
        self.controller = controller_REGISTRY[self.args.controller](self.args, self.env_scheme)
        self.buffer_scheme = self.controller.get_buffer_scheme()
        self.buffer = buffer_REGISTRY[self.args.buffer](self.args, self.buffer_scheme)
        self.test_buffer = buffer_REGISTRY[self.args.buffer](self.args, self.buffer_scheme)

    def reset(self, test_mode=False):
        current_env = self.test_env if test_mode else self.env
        reset_seed = None
        if getattr(self.args, "seed", None) is not None:
            if test_mode:
                reset_seed = self.args.seed + 1000000 + self.test_reset_count
                self.test_reset_count += 1
            else:
                reset_seed = self.args.seed + self.train_reset_count
                self.train_reset_count += 1

        current_env.reset(seed=reset_seed)
        self.t = 0

    def _start_episode(self, test_mode=False):
        # 开始一个episode
        episode_buffer = self.test_buffer if test_mode else self.buffer
        if test_mode:
            episode_buffer.clear()

        self.reset(test_mode=test_mode)
        episode_buffer.new_episode()

        self.current_test_mode = test_mode
        self.current_total_reward = 0
        current_env = self.test_env if test_mode else self.env
        self.current_state = current_env.get_state()
        self.current_terminated = current_env.terminated
        self.current_truncated = False
        self.episode_active = True
        self.episode_done = False

    def _finish_episode(self, test_mode=False):
        # 结束一个episode
        # 做收尾工作
        episode_buffer = self.test_buffer if test_mode else self.buffer

        episode_buffer.insert(self.current_state, "state", self.t)
        episode_buffer.insert(self.current_terminated, "terminated", self.t)
        episode_buffer.insert(self.current_truncated, "truncated", self.t)

        final_input = self.controller.builder.build_timestep(episode_buffer, self.t)
        episode_buffer.insert(final_input, "input", self.t)
        episode_buffer.finish_episode()

        total_reward = self.current_total_reward

        if test_mode:
            episode_buffer.clear()
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

    def start_test_phase(self):
        self.test_reset_count = 0

    def _write_current_step(self, batch_dict, key, item):
        target_slice = batch_dict[key][0, self.t]
        item_tensor = torch.as_tensor(item, dtype=target_slice.dtype, device=target_slice.device)
        target_slice.copy_(item_tensor.reshape_as(target_slice))
