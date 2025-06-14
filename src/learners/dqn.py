from copy import deepcopy
from torch import optim


class DQN:
    def __init__(self, args, scheme, controller, logger):
        self.args = args
        self.scheme = scheme
        self.controller = controller
        self.logger = logger

        # 准备好目标网络
        self.target_controller = deepcopy(self.controller)
        # 记录上次更新目标网络的时间
        self.last_target_update_episode = 0

        # 需要优化的参数和优化器
        self.param = list(self.controller.parameters())
        self.optimizer = optim.Adam(self.param, lr=self.args.lr)



    def learn(self, buffer, t_env, episode_num):
        # 获取基本的数据结构
        # terminated和states的长度比actions和rewards和filled多一个

        # 需要所有的states
        states = buffer["states"]
        # terminated是用来判断下一状态是否为结束状态的，所以截取时不需要第一个
        terminated = buffer["terminated"][:,1:].float().unsqueeze(-1)

        # 最后一个动作、reward和filled都是无效的，填充的数据而已
        actions = buffer["actions"][:, :-1]
        rewards = buffer["rewards"][:, :-1].unsqueeze(-1)
        filled = buffer["filled"][:, :-1].unsqueeze(-1)

        # 获取在线网络估计的Q值,注意取第一个step到倒数第二个step
        online_q = self.controller.forward(states[:,:-1])

        # 获取目标网络的估计Q值，注意取第二个step取到倒数最后一个step
        target_q = self.target_controller.forward(states[:,1:]).detach()

        # 获取所选动作的q值
        chosen_action_q_val = online_q.gather(2, actions)
        # 获取下一状态的最大q值动作的q值
        max_next_q_value = target_q.max(2)[0].unsqueeze(-1)

        # TD error, 几乎是所有q学习的关键更新指标
        td_error = chosen_action_q_val - (rewards + max_next_q_value * (1-terminated))

        # 忽略填充的步骤
        masked_td_error = td_error * filled

        loss = (masked_td_error**2).sum()/filled.sum()

        # 清空梯度，反向传播，执行更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 根据episode更新目标网络
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_target_network()
            self.last_target_update_episode = episode_num

    def _update_target_network(self):
        self.target_controller.agent.load_state_dict(self.controller.agent.state_dict())

    def cuda(self):
        self.controller.cuda()
        self.target_controller.cuda()

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

