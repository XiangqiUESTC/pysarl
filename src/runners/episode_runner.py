import torch

from .abstract_runner import AbstractRunner
import numpy as np


class EpisodeRunner(AbstractRunner):

    def __init__(self, args, logger):
        super().__init__(args, logger)

    # 这个函数会跑完一个完整的episode，收集整个episode的交互数据
    def run(self, test_mode=False):
        self.reset()
        terminated = self.env.terminated

        # 状态转移字典
        transaction_list = {
            "rewards": [],
            "states": [],
            "actions": [],
            "terminated": [],
        }

        while not terminated:
            # 选择动作？
            state = self.env.get_state()

            # 即使是一个state也要做batch化的处理，这是统一的要求
            batch_state = torch.unsqueeze(state, 0)
            actions = self.controller.select_action(batch_state, self.t_env, self.t, test_mode=test_mode)

            transaction_list["rewards"].append(state)
            transaction_list["actions"].append(actions[0])

            # 执行动作，actions是针对一个batch的state返回的所有行动的集合，所以这里要取actions[0]
            state, reward, done, info = self.env.step(actions[0])

            transaction_list["states"].append(reward)
            transaction_list["terminated"].append(done)

            self.t += 1

        transaction_tensor = {}

        for key, value in transaction_list.items():
            transaction_tensor[key] = torch.stack(value)

        self.t_env += self.t

        return transaction_tensor
