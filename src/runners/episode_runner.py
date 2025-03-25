from .abstract_runner import AbstractRunner
import numpy as np


class EpisodeRunner(AbstractRunner):

    def __init__(self, args, logger):
        super().__init__(args, logger)

    # 这个函数会跑完一个完整的episode，收集整个episode的交互数据
    def run(self, test_mode=False):
        self.reset()
        terminated = self.env.terminated

        while not terminated:
            # 选择动作？
            state = self.env.get_state()
            batch_state = np.expand_dims(state, axis=0)
            action = self.controller.select_action(batch_state, self.t_env, self.t, test_mode=test_mode)
            print(f"action是{action}")
            # 执行动作
            state, reward, terminated, info = self.env.step(action[0])

            print(reward)
            print(state)
            print(terminated)
            # 把交互数据放入replay buffer中

            self.t += 1

        self.t_env += self.t
        pass
