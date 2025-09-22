import torch

from .abstract_runner import AbstractRunner
import numpy as np


class EpisodeRunner(AbstractRunner):

    def __init__(self, args, logger):
        super().__init__(args, logger)

    # 这个函数会跑完一个完整的episode，收集整个episode的交互数据
    def run(self, test_mode=False):
        self.reset()
        terminated = truncated = self.env.terminated

        total_reward = 0

        # 状态转移字典 和一些必要的信息
        transaction = {
            "rewards": [],
            "states": [],
            "actions": [],
            "terminated": [],

            "filled": []
        }
        # 选择动作？
        state = self.env.get_state()
        while not (terminated or truncated):
            # 即使是一个state也要做batch化的处理，这是统一的要求
            batch_state = torch.unsqueeze(state, 0).to(self.args.device)
            actions = self.controller.select_action(batch_state, self.t_env, self.t, test_mode=test_mode)

            # 记录状态转移数据
            transaction["states"].append(state)
            transaction["actions"].append(actions)
            transaction["terminated"].append(torch.tensor([terminated]))
            # 记录步长
            transaction["filled"].append(torch.tensor([1]))

            # 执行动作，actions是针对一个batch的state返回的所有行动的集合，所以这里要取actions[0]
            state, reward, terminated, truncated, *_ = self.env.step(actions.item())

            # 累加奖励
            total_reward += reward

            transaction["rewards"].append(torch.tensor([reward]))

            self.t += 1
        # 记录终止信息
        transaction["states"].append(state)
        transaction["terminated"].append(torch.tensor([terminated]))

        self.episode += 1
        self.t_env += self.t

        self.logger.log_scalar("total_rewards", total_reward, self.t_env)

        if self.episode % self.args.log_interval == 0:
            self.logger.logger.info(f"Episode: {self.episode:>5} t_env: {self.t_env:>10} total_reward: {total_reward}", )
        return transaction
