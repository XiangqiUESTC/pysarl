import torch

from .abstract_runner import AbstractRunner


class EpisodeRunner(AbstractRunner):

    def __init__(self, args, logger):
        super().__init__(args, logger)
        # 每次运行只得到一个episode
        self.batch_size = 1

    # 这个函数会跑完一个完整的episode，收集整个episode的交互数据
    def run(self, test_mode=False):
        # 重置环境
        self.reset()
        # 新增一个episode
        self.buffer.new_episode()
        # 初始化终止符号和截断符号
        terminated = truncated = self.env.terminated
        # 记录reward
        total_reward = 0
        # 选择动作？
        state = self.env.get_state()
        while not (terminated or truncated):
            self.buffer.insert(state, "state", self.t)

            # 把列表类型的transaction数据包装成tensor返回，用于构建输入agent中的数据
            actions = self.controller.select_action(self.buffer, self.t_env, self.t, test_mode=test_mode)

            # 记录状态转移数据
            self.buffer.insert(actions[0], "action", self.t)
            self.buffer.insert(terminated, "terminated", self.t)

            # 记录步长
            self.buffer.insert(torch.tensor([1.0]), "filled", self.t)

            # 执行动作，actions是针对一个batch的state返回的所有行动的集合，所以这里要取actions[0]
            state, reward, terminated, truncated, *_ = self.env.step(actions.item())

            # 累加奖励
            total_reward += reward

            self.buffer.insert(reward, "rewards", self.t)

            self.t += 1
        # 记录终止信息
        self.buffer.insert(state, "states", self.t + 1)
        self.buffer.insert(terminated, "terminated", self.t + 1)

        self.episode += 1
        self.t_env += self.t

        self.logger.log_scalar("total_rewards", total_reward, self.t_env)

        if self.episode % self.args.log_interval == 0:
            self.logger.logger.info(f"Episode: {self.episode:>5} t_env: {self.t_env:>10} total_reward: {total_reward}", )

