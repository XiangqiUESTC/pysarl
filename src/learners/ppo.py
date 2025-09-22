from copy import deepcopy
from torch import optim
import numpy as np
import torch
import torch.nn.functional as F


class PPO:
    def __init__(self, args, scheme, controller, logger):
        self.args = args
        self.scheme = scheme
        self.controller = controller
        self.logger = logger

        """
        旧策略
        记录上次更长旧策略的时间
        """
        self.old_controller = deepcopy(self.controller)
        self.last_old_update_episode = 0

        # 需要优化的参数和优化器
        self.param = list(self.controller.parameters())
        self.optimizer = optim.Adam(self.param, lr=self.args.lr)

    def compute_returns(self, rewards, values, terminated):
        """
        计算广义优势函数 A_t
        ① 需要计算广义优势函数
        ② 不需要计算广义优势函数
        但是最终以 returns 的形式存在
        """
        returns = np.zeros_like(rewards)
        if self.args.gae:
            gae = 0
            for step in reversed(range(rewards.size(0))):
                # 跨 episode 时 gae 清零
                if terminated[step]:
                    gae = 0

                # 计算时序差分误差 δ = [r_t + γ(1-d)V(s_{t+1})] - V(s_t)
                delta = (
                    rewards[step]
                    + self.args.gamma * values[step + 1] * terminated[step + 1]
                    - values[step]
                )
                # 计算广义优势 A_t = δ + γ τ (1-d) A_{t+1}
                gae = (
                    delta
                    + self.args.gamma
                    * self.args.gae_lambda
                    * terminated[step + 1]
                    * gae
                )
                # 计算未来折扣回报 G_t ≈ Q(s_t, a_t) = A_t + V(s_t)
                returns[step] = gae + values[step]
        else:
            for step in reversed(range(rewards.size(0))):
                returns[step] = (
                    returns[step + 1] * self.args.gamma * terminated[step + 1]
                    + rewards[step]
                )
        return returns

    def learn(self, buffer, t_env, episode_num):
        batch = buffer.sample()
        # 获取基本的数据结构

        # terminated/states 的长度比 actions/rewards/filled 多一个
        # 需要所有的states
        old_states = batch["states"]
        # action/reward/filled 中的最后一个元素都是无效的，填充的数据而已
        old_actions = batch["actions"][:, :-1]
        old_rewards = batch["rewards"][:, :-1]
        # terminated 是用来判断下一状态是否为结束状态的，所以截取时不需要第一个
        old_terminated = batch["terminated"][:, 1:].float()
        old_filled = batch["filled"][:, :-1]

        old_logprobs = batch["logprobs"]
        old_state_values = batch["values"]

        # 计算未来回报
        returns = self.compute_returns(old_rewards, old_state_values, old_terminated)

        # 计算广义优势
        advantages = (
            self.compute_returns(old_rewards, old_state_values, old_filled)
            - old_state_values
        )

        for _ in range(self.args.K_epochs):
            # 当前 policy 网络计算动作的对数概率、状态价值、策略分布的熵
            logprobs, values, dist_entropy = self.controller.evaluate(
                old_states, old_actions
            )

            # 计算重要性比率
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算策略梯度
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            # 计算损失
            masked_action_loss = -torch.min(surr1, surr2) * old_filled
            action_loss = masked_action_loss.sum() / old_filled.sum()

            masked_value_loss = (returns - values) ** 2 * old_filled
            value_loss = masked_value_loss.sum() / old_filled.sum()

            loss = (
                action_loss
                + self.args.value_loss_weight * value_loss
                - self.args.entropy_loss_weight * dist_entropy
            )

            # 清空梯度，反向传播，执行更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新旧策略网络
        self._update_old_network()

        # 清空经验回放缓冲区
        buffer.clear()

    def _update_old_network(self):
        self.old_controller.agent.load_state_dict(self.controller.agent.state_dict())

    def cuda(self):
        self.controller.cuda()
        self.old_controller.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
