from copy import deepcopy
from torch import optim
import numpy as np
import torch
import torch.nn.functional as F
from components.agents.simple_critic import SimpleCritic
from utils.functions import compute_advantage


class PPO:
    def __init__(self, args, scheme, controller, logger):
        self.args = args
        self.scheme = scheme
        self.logger = logger

        # 演员 (策略网络)
        self.controller = controller
        # 评论员 (价值网络)
        self.critic = SimpleCritic(args, scheme)

        # 演员所需要优化的参数和优化器
        self.actor_param = list(self.controller.parameters())
        self.actor_optimizer = optim.Adam(self.actor_param, lr=self.args.actor_lr)
        # 评论员所需要优化的参数和优化器
        self.critic_param = list(self.critic.parameters())
        self.critic_optimizer = optim.Adam(self.critic_param, lr=self.args.critic_lr)

    def learn(self, buffer, t_env, episode_num):
        batch = buffer.sample()
        # 获取基本的数据结构

        # terminated/states 的长度比 actions/rewards/filled 多一个
        # 需要所有的 states
        states = batch["states"]
        # action/reward/filled 中的最后一个元素都是无效的，填充的数据而已
        actions = batch["actions"][:, :-1]
        rewards = batch["rewards"][:, :-1]
        # terminated 是用来判断下一状态是否为结束状态的，所以截取时不需要第一个
        terminated = batch["terminated"][:, 1:].float()
        filled = batch["filled"][:, :-1]

        # (为评论员) 计算 TD 目标
        td_target = rewards + self.args.gamma * self.critic(states[:, 1:]) * (
            1 - terminated
        )
        # (为评论员) 计算时序差分误差
        td_delta = td_target - self.critic(states[:, :-1])

        # (为演员) 计算旧的动作对数概率
        old_log_probs = torch.log(
            self.controller.forward(states).gather(1, actions)
        ).detach()
        # (为演员) 计算广义优势
        advantages = compute_advantage(
            self.args.gamma, self.args.lmbda, td_delta.cpu()
        ).to(self.args.device)

        # 使用当前 batch 更新 epochs 次网络参数
        for _ in range(self.args.epochs):
            # 当前 policy 网络计算动作的对数概率、状态价值、策略分布的熵
            log_probs = torch.log(self.controller.forward(states).gather(1, actions))

            # 计算重要性比率
            ratios = torch.exp(log_probs - old_log_probs)

            # 计算策略梯度
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(
                    ratios, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon
                )
                * advantages
            )

            # 计算演员的损失
            masked_action_loss = -torch.min(surr1, surr2) * filled
            actor_loss = masked_action_loss.sum() / filled.sum()
            # 计算评论员的损失
            # td_target 需要 .detach()，否则会导致二次梯度传播
            masked_value_loss = (
                td_target.detach() - self.critic(states[:, :-1])
            ) ** 2 * filled
            critic_loss = masked_value_loss.sum() / filled.sum()

            # 清空梯度、反向传播、执行更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 更新旧策略网络
        self._update_old_network()

        # 清空经验回放缓冲区
        buffer.clear()

    def _update_old_network(self):
        pass

    def cuda(self):
        self.controller.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
