from copy import deepcopy

import torch
from torch import optim
from components.agents import REGISTRY


class Reinforce:
    def __init__(self, args, scheme, controller, logger):
        self.args = args
        self.scheme = scheme
        self.controller = controller
        self.logger = logger

        if args.baseline:
            self.critic = REGISTRY[args.critic.name](args.critic, scheme)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.optimizer = optim.Adam(self.controller.parameters(), lr=args.lr)


    def learn(self, buffer, t_env, episode_num):
        batch = buffer.sample()
        # 获取基本的数据结构
        # terminated和states的长度比actions和rewards和filled多一个

        # 需要所有的states
        states = batch["states"] # batch_size × (episode_length+1) × state_dim

        # 最后一个动作、reward和filled都是无效的，填充的数据而已
        actions = batch["actions"][:, :-1] # batch_size × episode_length × 1
        rewards = batch["rewards"][:, :-1] # batch_size × episode_length × 1
        filled = batch["filled"][:, :-1] # batch_size × episode_length × 1

        # 获取选择动作的概率
        action_probs = self.controller.forward(states[:,:-1])
        chosen_action_probs = torch.gather(action_probs, -1, actions)

        # 根据formula选择3种不同的公式
        if self.args.formula == 1:
            episode_rewards = (rewards * filled).sum(dim=1).squeeze(-1) # batch_size
            log_chosen_action_probs_sum = torch.log(chosen_action_probs).sum(dim=1).squeeze(-1) # batch_size
            loss = -(episode_rewards * log_chosen_action_probs_sum).sum()/filled.sum()
        elif self.args.formula == 2:
            log_chosen_action_probs_cumsum = torch.log(chosen_action_probs).cumsum(dim=1)
            loss = -(rewards*log_chosen_action_probs_cumsum).sum()/filled.sum()
        elif self.args.formula == 3:
            log_chosen_action_probs = torch.log(chosen_action_probs)
            return_sample = (rewards * filled).flip(1).cumsum(dim=1).flip(1)
            if self.args.baseline:
                state_vals = self.critic(states[:,:-1]).detach().clone()
                loss = -(log_chosen_action_probs * (return_sample-state_vals) * filled).sum()/filled.sum()
            else:
                loss = -(log_chosen_action_probs * return_sample * filled).sum()/filled.sum()
        else:
            raise NotImplementedError("Method REINFORCE only has 3 kinds of formulas!")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        buffer.clear()

        # 如果有基线还要优化基线
        # 以return_sample作为估计
        if self.args.baseline:
            return_sample = (rewards * filled).flip(1).cumsum(dim=1).flip(1)
            v_values = self.critic(states[:,:-1])
            value_loss = ((return_sample - v_values)**2*filled).sum()/filled.sum()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()


    def cuda(self):
        self.controller.cuda()
        if self.args.baseline:
            self.critic.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

